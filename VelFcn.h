#ifndef included_VelFcn
#define included_VelFcn

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibtk/CartGridFunction.h>
#include <ibtk/ibtk_utilities.h>

#include <CartesianGridGeometry.h>

#include <armadillo>

/////////////////////////////// CLASS DEFINITION /////////////////////////////

/*!
 * \brief Method to initialize the value of the advection velocity u.
 */
class VelFcn : public IBTK::CartGridFunction
{
public:
    /*!
     * \brief Constructor.
     */
    VelFcn(std::string object_name,
           SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
           SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db);

    /*!
     * \brief Destructor.
     */
    ~VelFcn();

    /*!
     * Indicates whether the concrete CartGridFunction object is time dependent.
     */
    bool isTimeDependent() const
    {
        return true;
    }

    /*!
     * Set the data on the patch interior to some initial values.
     */
    void setDataOnPatch(int data_idx,
                        SAMRAI::tbox::Pointer<SAMRAI::hier::Variable<NDIM>> var,
                        SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch,
                        double data_time,
                        bool initial_time = false,
                        SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>> level =
                            SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>>(NULL));

    arma::mat evaluateVelocity(double t, const arma::mat& X);

protected:
private:
    /*!
     * \brief Default constructor.
     *
     * \note This constructor is not implemented and should not be used.
     */
    VelFcn();

    /*!
     * \brief Copy constructor.
     *
     * \note This constructor is not implemented and should not be used.
     *
     * \param from The value to copy to this object.
     */
    VelFcn(const VelFcn& from);

    /*!
     * \brief Assignment operator.
     *
     * \note This operator is not implemented and should not be used.
     *
     * \param that The value to assign to this object.
     *
     * \return A reference to this object.
     */
    VelFcn& operator=(const VelFcn& that);

    /*!
     * Read input values, indicated above, from given database.
     */
    void getFromInput(SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> db);

    double d_yup = std::numeric_limits<double>::quiet_NaN(), d_ylow = std::numeric_limits<double>::quiet_NaN();

    double d_mu = std::numeric_limits<double>::quiet_NaN(), d_gradp = std::numeric_limits<double>::quiet_NaN();

    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_u_var;
    int d_u_idx = IBTK::invalid_index;
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> d_hierarchy;

    // Datatype for communicating point and index
    MPI_Datatype d_pt_spec_type;
};

/////////////////////////////// INLINE ///////////////////////////////////////

// #include "VelFcn.I"

//////////////////////////////////////////////////////////////////////////////

#endif // #ifndef included_VelFcn
