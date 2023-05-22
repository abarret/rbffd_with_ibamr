#include <ibamr/INSCollocatedHierarchyIntegrator.h>
#include <ibamr/INSStaggeredHierarchyIntegrator.h>
#include <ibamr/app_namespaces.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/IBTK_MPI.h>
#include <ibtk/muParserCartGridFunction.h>
#include <ibtk/muParserRobinBcCoefs.h>

#include "KernelPack/base/DomainDescriptor.h"
#include "KernelPack/solvers/RBFSLADSolver.h"
#include "VelFcn.h"

#include <petscsys.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <LoadBalancer.h>
#include <SAMRAI_config.h>
#include <StandardTagAndInitialize.h>
#include <omp.h>

using namespace KernelPack;

arma::vec
initConc(const double time, const arma::vec& X)
{
    return 0 * X;
}

arma::mat
computeVel(const double time, const arma::vec& X, VelFcn& fcn)
{
    return fcn.evaluateVelocity(time, X);
}

/*******************************************************************************
 * For each run, the input filename and restart information (if needed) must   *
 * be given on the command line.  For non-restarted case, command line is:     *
 *                                                                             *
 *    executable <input file name>                                             *
 *                                                                             *
 * For restarted run, command line is:                                         *
 *                                                                             *
 *    executable <input file name> <restart directory> <restart number>        *
 *                                                                             *
 *******************************************************************************/
int
main(int argc, char* argv[])
{
    // Initialize IBAMR and libraries. Deinitialization is handled by this object
    // as well.
    IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);

    int num_openmp_threads = 1;
    omp_set_num_threads(num_openmp_threads);

    { // cleanup dynamically allocated objects prior to shutdown

        // Parse command line options, set some standard options from the input
        // file, initialize the restart database (if this is a restarted run),
        // and enable file logging.
        Pointer<AppInitializer> app_initializer = new AppInitializer(argc, argv, "INS.log");
        Pointer<Database> input_db = app_initializer->getInputDatabase();

        // Get various standard options set in the input file.
        const bool dump_viz_data = app_initializer->dumpVizData();
        const int viz_dump_interval = app_initializer->getVizDumpInterval();
        const bool uses_visit = dump_viz_data && app_initializer->getVisItDataWriter();

        const bool dump_restart_data = app_initializer->dumpRestartData();
        const int restart_dump_interval = app_initializer->getRestartDumpInterval();
        const string restart_dump_dirname = app_initializer->getRestartDumpDirectory();

        const bool dump_timer_data = app_initializer->dumpTimerData();
        const int timer_dump_interval = app_initializer->getTimerDumpInterval();

        // Create major algorithm and data objects that comprise the
        // application.  These objects are configured from the input database
        // and, if this is a restarted run, from the restart database.
        Pointer<CartesianGridGeometry<NDIM>> grid_geometry = new CartesianGridGeometry<NDIM>(
            "CartesianGeometry", app_initializer->getComponentDatabase("CartesianGeometry"));
        Pointer<PatchHierarchy<NDIM>> patch_hierarchy = new PatchHierarchy<NDIM>("PatchHierarchy", grid_geometry);
        Pointer<StandardTagAndInitialize<NDIM>> error_detector = new StandardTagAndInitialize<NDIM>(
            "StandardTagAndInitialize", nullptr, app_initializer->getComponentDatabase("StandardTagAndInitialize"));
        Pointer<BergerRigoutsos<NDIM>> box_generator = new BergerRigoutsos<NDIM>();
        Pointer<LoadBalancer<NDIM>> load_balancer =
            new LoadBalancer<NDIM>("LoadBalancer", app_initializer->getComponentDatabase("LoadBalancer"));
        Pointer<GriddingAlgorithm<NDIM>> gridding_algorithm =
            new GriddingAlgorithm<NDIM>("GriddingAlgorithm",
                                        app_initializer->getComponentDatabase("GriddingAlgorithm"),
                                        error_detector,
                                        box_generator,
                                        load_balancer);

        // Create initial condition specification objects.
        Pointer<VelFcn> u_fcn = new VelFcn("u_fcn", patch_hierarchy, app_initializer->getComponentDatabase("VelFcn"));

        // Get visualization object
        Pointer<VisItDataWriter<NDIM>> visit_data_writer = app_initializer->getVisItDataWriter();

        // Create boundary condition specification objects (when necessary).
        const IntVector<NDIM>& periodic_shift = grid_geometry->getPeriodicShift();
        vector<RobinBcCoefStrategy<NDIM>*> u_bc_coefs(NDIM, nullptr);

        // Create velocity variable and context pair
        auto var_db = VariableDatabase<NDIM>::getDatabase();
        Pointer<CellVariable<NDIM, double>> u_var = new CellVariable<NDIM, double>("U", NDIM);
        const int u_idx = var_db->registerVariableAndContext(u_var, var_db->getContext("Context"), 1 /*gcw*/);

        // Initialize hierarchy configuration and data on all patches.
        int tag_buffer = input_db->getInteger("TAG_BUFFER");
        int ln = 0;
        bool done = false;
        while (!done && gridding_algorithm->levelCanBeRefined(ln))
        {
            gridding_algorithm->makeFinerLevel(patch_hierarchy, 0.0, true, tag_buffer);
            done = !patch_hierarchy->finerLevelExists(ln);
            ++ln;
        }

        // We need to create the domain for the RBFSLADSolver.
        DomainDescriptor domain;

        // Generate points around the boundary.
        pout << "Generating the advection diffusion domain!\n";
        Pointer<Database> adv_diff_grid_db = input_db->getDatabase("AdvDiffGrid");
        const double dx = adv_diff_grid_db->getDouble("dx_adv_diff");
        VectorNd xlow, L;
        adv_diff_grid_db->getDoubleArray("l", L.data(), NDIM);
        adv_diff_grid_db->getDoubleArray("xlow", xlow.data(), NDIM);
        int tot_pts = 0;
        std::array<unsigned int, NDIM> num_pts;
        for (int d = 0; d < NDIM; ++d)
        {
            num_pts[d] = static_cast<int>((L[d] - xlow[d]) / dx);
            tot_pts += 2 * num_pts[d];
        }
        arma::mat bdry_pts(tot_pts - 4, NDIM);
        int shft = 0;
        // X points
        for (int i = 0; i < num_pts[0]; ++i)
        {
            bdry_pts(i + shft, 0) = xlow[0] + static_cast<double>(i) * dx;
            bdry_pts(i + shft, 1) = xlow[1];
            // Two boundaries
            bdry_pts(i + shft + num_pts[0], 0) = xlow[0] + static_cast<double>(i) * dx;
            bdry_pts(i + shft + num_pts[0], 1) = xlow[1] + L[1];
        }
        shft += 2 * num_pts[0];
        for (int i = 1; i < num_pts[1] - 1; ++i)
        {
            bdry_pts(i + shft - 1, 0) = xlow[0];
            bdry_pts(i + shft - 1, 1) = xlow[1] + static_cast<double>(i) * dx;
            // Two boundaries
            bdry_pts(i + shft + num_pts[1] - 3, 0) = xlow[0] + L[0];
            bdry_pts(i + shft + num_pts[1] - 3, 1) = xlow[1] + static_cast<double>(i) * dx;
        }
        pout << bdry_pts << "\n";
        const int rank = IBTK_MPI::getRank();
        pout << "Calling on rank 0\n";
        if (rank == 0)
        {
            // Note only the first rank needs it does not use MPI.
            pout << "Generating sampling params\n";
            DomainDescriptor::SamplingParams sampleParams;
            sampleParams.radius = dx;
            sampleParams.doOuterRefinement = 0;
            sampleParams.outerFractionOfh = 0.75;
            sampleParams.outerRefinementZoneSizeAsMultipleOfh = 2.0;
            sampleParams.pdsSurfSupersampleFac = 8;
            pout << "Calling generateSmoothDomainNodes\n";
            domain.generateSmoothDomainNodes(bdry_pts, sampleParams, num_openmp_threads);
        }
        pout << "Finished generating domain.\n";
        // Introduce a barrier to ensure all other processors are avaible for generating the domain
        IBTK_MPI::barrier();

        // Now create the solver.
        RBFSLADSolver solver;

        if (rank == 0)
        {
            const int num_bdry_nodes = domain.getNumBdryNodes();
            std::vector<double> a(num_bdry_nodes, 1.0), b(num_bdry_nodes, 0.0);
            solver.initParallel(
                domain, 3, input_db->getDouble("MAX_DT"), input_db->getDouble("DIFF_COEF"), a, b, num_openmp_threads);
        }
        // Introduce a barrier to ensure all other processors are available for generating the solver
        IBTK_MPI::barrier();

        if (rank == 0)
        {
            arma::vec init_conc = initConc(0.0 /*time*/, domain.getIntBdryNodes());
            auto forcing_fcn = [](double, double, arma::mat& X) -> arma::vec
            {
                arma::vec force(X.n_cols, 0.0);
                return force;
            };
            auto bc_fcn = [](double, arma::mat& nr, double, arma::mat& X) -> arma::vec
            {
                arma::vec bdry_vals(X.n_cols, 0.0);
                return bdry_vals;
            };
            auto velFcn = [&u_fcn](double t, arma::mat X) -> arma::mat { return computeVel(t, X, *u_fcn); };
            velFcn(0.0, domain.getInteriorNodes());
            //            solver.solveBDF3Parallel(input_db->getDouble("END_TIME"), init_conc, velFcn,
            //            forcing_fcn/*forcing fcn*/, bc_fcn/*bc*/, num_openmp_threads);
        }
        // Barrier so that all other processors are available.
        IBTK_MPI::barrier();

        // Print the input database contents to the log file.
        plog << "Input database:\n";
        input_db->printClassData(plog);

        // Write out initial visualization data.
        if (dump_viz_data && uses_visit)
        {
            pout << "\n\nWriting visualization files...\n\n";
            visit_data_writer->writePlotData(patch_hierarchy, 0, 0.0);
        }

        // Main time step loop.

        // Cleanup boundary condition specification objects (when necessary).
        for (unsigned int d = 0; d < NDIM; ++d) delete u_bc_coefs[d];

    } // cleanup dynamically allocated objects prior to shutdown
} // main
