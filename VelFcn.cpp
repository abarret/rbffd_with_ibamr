// ---------------------------------------------------------------------
//
// Copyright (c) 2014 - 2020 by the IBAMR developers
// All rights reserved.
//
// This file is part of IBAMR.
//
// IBAMR is free software and is distributed under the 3-clause BSD
// license. The full text of the license can be found in the file
// COPYRIGHT at the top level directory of IBAMR.
//
// ---------------------------------------------------------------------

#include <ibamr/app_namespaces.h>

#include <ibtk/HierarchyGhostCellInterpolation.h>
#include <ibtk/IBTK_CHKERRQ.h>
#include <ibtk/IBTK_MPI.h>
#include <ibtk/IndexUtilities.h>

#include "VelFcn.h"

#include <SAMRAI_config.h>

/////////////////////////////// STATIC ///////////////////////////////////////
static int gcw = 1;

/*!
 *  PtSpec is an MPI compatible data type that contains a list of doubles and a integer. The constructor is templated
 * over the type Point which must provide the operator() which gives the location of the first index of an array of NDIM
 * doubles.
 */
struct PtSpec
{
    PtSpec()
    {
    }
    template <typename Point>
    PtSpec(const Point& pt, const int dof) : d_dof(dof)
    {
        for (int d = 0; d < NDIM; ++d) d_pt[d] = pt(d);
        // intentionally blank
    }

    friend ostream& operator<<(ostream& os, const PtSpec& pt);

    std::array<double, NDIM> d_pt;
    int d_dof = -1;
};

ostream&
operator<<(ostream& os, const PtSpec& pt)
{
    for (int d = 0; d < NDIM - 1; ++d) os << pt.d_pt[d] << ", ";
    os << pt.d_pt[NDIM - 1];
    return os;
}
/////////////////////////////// PUBLIC ///////////////////////////////////////

VelFcn::VelFcn(std::string object_name, Pointer<PatchHierarchy<NDIM>> hierarchy, Pointer<Database> input_db)
    : CartGridFunction(std::move(object_name)),
      d_u_var(new CellVariable<NDIM, double>(d_object_name + "::Velocity", NDIM)),
      d_hierarchy(hierarchy)
{
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    d_u_idx = var_db->registerVariableAndContext(
        d_u_var, var_db->getContext(d_object_name + "::Context"), IntVector<NDIM>(gcw) /*gcw*/);
    // Create a patch index for the velocity
    getFromInput(input_db);

    // Create the MPI datatype.
    PtSpec pt_spec;
    std::array<MPI_Datatype, 2> types = { MPI_DOUBLE, MPI_INT };
    std::array<int, 2> block_length = { NDIM, 1 };
    std::array<MPI_Aint, 2> disp = {
        reinterpret_cast<std::uintptr_t>(&pt_spec.d_pt) - reinterpret_cast<std::uintptr_t>(&pt_spec),
        reinterpret_cast<std::uintptr_t>(&pt_spec.d_dof) - reinterpret_cast<std::uintptr_t>(&pt_spec)
    };
    MPI_Type_create_struct(2, block_length.data(), disp.data(), types.data(), &d_pt_spec_type);
    MPI_Type_commit(&d_pt_spec_type);
    return;
} // VelFcn

VelFcn::~VelFcn()
{
    MPI_Type_free(&d_pt_spec_type);
    return;
} // ~VelFcn

void
VelFcn::setDataOnPatch(const int data_idx,
                       Pointer<Variable<NDIM>> /*var*/,
                       Pointer<Patch<NDIM>> patch,
                       const double data_time,
                       const bool initial_time,
                       Pointer<PatchLevel<NDIM>> /*level*/)
{
    Pointer<CellData<NDIM, double>> u_data = patch->getPatchData(data_idx);
    TBOX_ASSERT(u_data);
    if (initial_time)
    {
        u_data->fillAll(0.0);
        return;
    }

    Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
    const double* const dx = pgeom->getDx();
    const double* const xlow = pgeom->getXLower();
    const hier::Index<NDIM>& idx_low = patch->getBox().lower();

    // Fill in velocity
    // For now use poiseulle flow
    for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++)
    {
        const CellIndex<NDIM> idx = ci();
        VectorNd x;
        for (int d = 0; d < NDIM; ++d) x[d] = xlow[d] + dx[d] * (static_cast<double>(idx(d) - idx_low(d)) + 0.5);
        (*u_data)(idx, 0) = d_gradp / (2.0 * d_mu) * (x[1] - d_ylow) * (x[1] - d_yup);
    }
    return;
} // setDataOnPatch

arma::mat
VelFcn::evaluateVelocity(const double t, const arma::mat& X)
{
    int ierr;
    // First fill in the velocity
    const int coarsest_ln = 0;
    const int finest_ln = d_hierarchy->getFinestLevelNumber();
    Pointer<CartesianGridGeometry<NDIM>> grid_geom = d_hierarchy->getGridGeometry();
    setDataOnPatchHierarchy(d_u_idx, d_u_var, d_hierarchy, t, false, coarsest_ln, finest_ln);

    // Fill ghost cells on the patch hierarchy
    using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    ITC ghost_cell_comp(
        d_u_idx, "CONSERVATIVE_LINEAR_REFINE", false, "NONE", "LINEAR", false, nullptr, nullptr, "LINEAR");
    HierarchyGhostCellInterpolation ghost_cell_fill;
    ghost_cell_fill.initializeOperatorState(ghost_cell_comp, d_hierarchy, coarsest_ln, finest_ln);
    ghost_cell_fill.fillData(t);

    // In order to interpolate velocity to points, we need to determine which point is located in a given patch.
    // To prepare for this, we get the patch boxes of all patches on every processor, and send them to the root process.
    const int rank = IBTK_MPI::getRank();
    const int nodes = IBTK_MPI::getNodes();
    std::vector<std::vector<std::vector<Box<NDIM>>>> patch_boxes_ln(finest_ln - coarsest_ln + 1);
    std::vector<std::vector<unsigned int>> num_boxes_ln(finest_ln - coarsest_ln + 1);
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        patch_boxes_ln[ln].resize(rank);
        num_boxes_ln[ln].resize(nodes, 0);
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            patch_boxes_ln[ln][rank].push_back(patch->getBox());
            num_boxes_ln[ln][rank]++;
        }

        // Now send each box to the root process.
        // First the number of boxes so we can allocate space.
        IBTK_MPI::allToOneSumReduction(
            reinterpret_cast<int*>(num_boxes_ln[ln].data()), num_boxes_ln[ln].size(), 0 /*root*/);
        if (rank == 0)
        {
            for (int i = 1; i < nodes; ++i) patch_boxes_ln[ln][i].resize(num_boxes_ln[ln][i]);
        }

        // Second, send the actual boxes to the root processor.
        // A box consists of two NDIM IntVectors. We'll send the two IntVectors and create a box locally.
        if (rank != 0)
        {
            for (const auto& patch_box : patch_boxes_ln[ln][rank])
            {
                IBTK_MPI::send(&patch_box.lower()(0), NDIM, 0, false, 1);
                IBTK_MPI::send(&patch_box.upper()(0), NDIM, 0, false, 2);
            }
        }
        else
        {
            for (int i = 1; i < nodes; ++i)
            {
                for (int box_num = 0; box_num < num_boxes_ln[ln][i]; ++box_num)
                {
                    // Throwaway argument for required reference
                    int size = NDIM;
                    IBTK_MPI::recv(&patch_boxes_ln[ln][i][box_num].lower()(0), size, i, false, 1);
                    IBTK_MPI::recv(&patch_boxes_ln[ln][i][box_num].upper()(0), size, i, false, 2);
                }
            }
        }
    }

    // At this point, the root process has all the boxes in the domain. We can now sort points according to their box.
    std::vector<std::vector<std::vector<PtSpec>>> sorted_pts_ln;
    if (rank == 0)
    {
        sorted_pts_ln.resize(finest_ln, std::vector<std::vector<PtSpec>>(nodes));
        // OpenMP-ize this loop
        for (int pt_num = 0; pt_num < X.n_cols; ++pt_num)
        {
            bool found_home = false;
            // Start searching on finest level
            for (int ln = finest_ln; ln >= coarsest_ln && !found_home; --ln)
            {
                Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
                const hier::Index<NDIM>& idx =
                    IndexUtilities::getCellIndex(X.col(pt_num), grid_geom, level->getRatio());
                // Now check whether this index is inside the given boxes
                for (int i = 0; i < nodes && !found_home; ++i)
                {
                    for (const auto& box : patch_boxes_ln[ln][i])
                    {
                        if (box.contains(idx))
                        {
                            sorted_pts_ln[ln][i].push_back(PtSpec(X.col(pt_num), pt_num));
                            found_home = true;
                            break;
                        }
                    }
                }
            }
        }
    }
    IBTK_MPI::barrier();

    // We have all points sorted according to their process and level. We now need to send this information back to each
    // process
    std::vector<std::vector<PtSpec>> interp_pts(finest_ln - coarsest_ln + 1);
    std::vector<int> num_interp_pts(finest_ln - coarsest_ln + 1);
    if (rank == 0)
    {
        for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
        {
            for (int i = 1; i < nodes; ++i)
            {
                unsigned int size = sorted_pts_ln[ln][i].size();
                IBTK_MPI::send(&size, 1, i, false);
            }
        }
    }
    else
    {
        for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
        {
            unsigned int size;
            // Dummy argument for required reference
            int num_size = 1;
            IBTK_MPI::recv(&size, num_size, 0 /*sending proc*/, false);
            interp_pts[ln].resize(size);
        }
    }

    // We have the correct sizes on every processor. Now send the actual data.
    if (rank == 0)
    {
        for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
        {
            for (int i = 1; i < nodes; ++i)
            {
                // Send the point to the point's processor
                MPI_Send(sorted_pts_ln[ln][i].data(),
                         sorted_pts_ln[ln][i].size(),
                         d_pt_spec_type,
                         i,
                         i,
                         IBTK_MPI::getCommunicator());
            }
        }
    }
    else
    {
        for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
        {
            MPI_Recv(interp_pts[ln].data(),
                     num_interp_pts[ln],
                     d_pt_spec_type,
                     0,
                     rank,
                     IBTK_MPI::getCommunicator(),
                     MPI_STATUS_IGNORE);
        }
    }

    // Now each processor has it's set of points. Find the velocity for each set.
    std::vector<std::vector<PtSpec>> vel_pts(finest_ln - coarsest_ln + 1);
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        vel_pts[ln].resize(interp_pts[ln].size());
        for (size_t i = 0; i < vel_pts[ln].size(); ++i)
        {
            const hier::Index<NDIM>& idx =
                IndexUtilities::getCellIndex(interp_pts[ln][i].d_pt, grid_geom, level->getRatio());
            // Determine the patch
            bool found_patch = false;
            for (PatchLevel<NDIM>::Iterator p(level); p && found_patch; p++)
            {
                Pointer<Patch<NDIM>> patch = level->getPatch(p());
                const Box<NDIM>& box = patch->getBox();
                if (box.contains(idx))
                {
                    found_patch = true;
                    // Now interpolate the data
                    Pointer<CellData<NDIM, double>> u_data = patch->getPatchData(d_u_idx);
                    Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
                    const double* const dx = pgeom->getDx();
                    const double* const xlow = pgeom->getXLower();
                    const hier::Index<NDIM>& idx_low = box.lower();
                    VectorNd xij;
                    for (int d = 0; d < NDIM; ++d)
                        xij[d] = xlow[d] + dx[d] * (static_cast<double>(idx(d) - idx_low(d)) + 0.5);
                    // Quadratic interpolant in each direction
                    VectorNd vel;
                    for (int d = 0; d < NDIM; ++d)
                    {
                        IntVector<NDIM> dir = 0;
                        dir(d) = 1;
                        const double alpha = interp_pts[ln][i].d_pt[d] - xij[d];
                        vel(d) = (*u_data)(idx) * (1.0 - alpha - alpha * (alpha - dx[d])) +
                                 (*u_data)(idx + dir) * (alpha + 0.5 * alpha * (alpha - dx[d])) +
                                 (*u_data)(idx - dir) * 0.5 * alpha * (alpha - dx[d]);
                    }
                    vel_pts[ln][i] = PtSpec(vel, interp_pts[ln][i].d_dof);
                }
            }
            // Something has gone wrong if we haven't found the patch.
            if (!found_patch) TBOX_ERROR("Could not find the patch for point " << interp_pts[ln][i] << "\n");
            TBOX_ASSERT(found_patch);
        }
    }

    // Now each processor has it's own velocity value. Need to send it back to the root processor.
    std::vector<std::vector<std::vector<PtSpec>>> global_vel_pts;
    if (rank == 0)
    {
        global_vel_pts.resize(finest_ln - coarsest_ln + 1);
        // First vector is stored in vel_pts. Move it.
        global_vel_pts[0] = std::move(vel_pts);
        // Now recieve each other processors list of points
        for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
        {
            global_vel_pts[ln].resize(nodes);
            for (int i = 1; i < nodes; ++i)
            {
                global_vel_pts[ln][i].resize(sorted_pts_ln[ln][i].size());
                MPI_Recv(global_vel_pts[ln][i].data(),
                         global_vel_pts[ln][i].size(),
                         d_pt_spec_type,
                         i,
                         i,
                         IBTK_MPI::getCommunicator(),
                         MPI_STATUS_IGNORE);
            }
        }
    }
    else
    {
        for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
        {
            MPI_Send(vel_pts[ln].data(), vel_pts[ln].size(), d_pt_spec_type, 0, rank, IBTK_MPI::getCommunicator());
        }
    }

    // Barrier so that processors are free.
    IBTK_MPI::barrier();

    // Now fill in the actual velocity vector
    // OpenMP this loop? How??
    arma::vec vel;
    if (rank == 0)
    {
        vel = 0 * X;
        for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
        {
            for (const auto& vel_pts : global_vel_pts[ln])
            {
                for (const auto& vel_pt : vel_pts)
                {
                    for (int d = 0; d < NDIM; ++d) vel.col(vel_pt.d_dof)[d] = vel_pt.d_pt[d];
                }
            }
        }
    }

    // Put a barrier here in case we parallel the above loop
    // IBTK_MPI::barrier();
    return vel;
}

/////////////////////////////// PRIVATE //////////////////////////////////////

void
VelFcn::getFromInput(Pointer<Database> db)
{
    TBOX_ASSERT(db);
    if (db)
    {
        d_ylow = db->getDouble("ylow");
        d_yup = db->getDouble("yup");
        d_mu = db->getDouble("mu");
        d_gradp = db->getDouble("gradp");
    }
    return;
} // getFromInput

//////////////////////////////////////////////////////////////////////////////
