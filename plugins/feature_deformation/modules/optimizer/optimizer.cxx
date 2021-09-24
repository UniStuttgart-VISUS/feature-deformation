#include "optimizer.h"

#include "../feature_deformation/curvature.h"
#include "../feature_deformation/gradient.h"
#include "../feature_deformation/grid.h"

#include "vtkAlgorithm.h"
#include "vtkDoubleArray.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkSmartPointer.h"
#include "vtkStructuredGrid.h"

#include <array>
#include <iostream>

vtkStandardNewMacro(optimizer);

optimizer::optimizer()
{
    this->SetNumberOfInputPorts(2);
    this->SetNumberOfOutputPorts(1);
}

optimizer::~optimizer()
{
}

int optimizer::RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector* output_vector)
{
    return 1;
}

int optimizer::FillInputPortInformation(int port, vtkInformation* info)
{
    if (port == 0)
    {
        info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkStructuredGrid");
        return 1;
    }
    else if (port == 1)
    {
        info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkStructuredGrid");
        return 1;
    }

    return 0;
}

int optimizer::RequestData(vtkInformation* vtkNotUsed(request), vtkInformationVector** input_vector, vtkInformationVector* output_vector)
{
    // Get input grid and data
    auto original_grid = vtkStructuredGrid::SafeDownCast(input_vector[0]->GetInformationObject(0)->Get(vtkDataObject::DATA_OBJECT()));
    auto deformed_grid = vtkStructuredGrid::SafeDownCast(input_vector[1]->GetInformationObject(0)->Get(vtkDataObject::DATA_OBJECT()));

    auto vector_field_original = GetInputArrayToProcess(0, original_grid);
    auto vector_field_deformed = GetInputArrayToProcess(1, deformed_grid);
    auto jacobian_field = GetInputArrayToProcess(2, deformed_grid);

    if (vector_field_original == nullptr || vector_field_deformed == nullptr || jacobian_field == nullptr)
    {
        std::cerr << "All input fields must be provided." << std::endl;
        return 0;
    }

    const grid original_vector_field(original_grid, vector_field_original);
    const grid deformed_vector_field(deformed_grid, vector_field_deformed, jacobian_field);

    // Calculate initial gradient difference
    auto original_curvature = curvature_and_torsion(original_vector_field);
    auto deformed_curvature = curvature_and_torsion(deformed_vector_field);

    auto gradient_difference = vtkSmartPointer<vtkDoubleArray>::New();
    gradient_difference->SetNumberOfComponents(original_curvature.curvature_gradient->GetNumberOfComponents());
    gradient_difference->SetNumberOfTuples(original_curvature.curvature_gradient->GetNumberOfTuples());

    for (vtkIdType i = 0; i < original_curvature.curvature_gradient->GetNumberOfTuples(); ++i)
    {
        gradient_difference->SetValue(i,
            deformed_curvature.curvature_gradient->GetComponent(i, 0) - original_curvature.curvature_gradient->GetComponent(i, 0));
    }

    // Calculate initial min, max error
    auto error_min = gradient_difference->GetDataTypeValueMin();
    auto error_max = gradient_difference->GetDataTypeValueMax();

    // Iterative optimization
    bool converged = false;

    auto vector_block = vtkSmartPointer<vtkDoubleArray>::New();
    vector_block->SetNumberOfComponents(3);
    vector_block->SetNumberOfTuples(25); // TODO: 3D

    Eigen::Matrix<double, 50, 1> gradient; // TODO: 3D

    for (int step = 0; step < NumSteps && !converged; ++step)
    {
        // For each 5x5(x5) block of nodes, calculate partial derivatives of the
        // curvature gradient difference in direction of the degrees of freedom.
        // Use gradient descent to perform a single step for respective center
        // vertex, minimizing its curvature gradient difference.
        for (int z = 0; z < original_vector_field.dimensions()[2]; ++z)
        {
            for (int y = 0; y < original_vector_field.dimensions()[1]; ++y)
            {
                for (int x = 0; x < original_vector_field.dimensions()[0]; ++x)
                {
                    // Create grid block
                    for (int zz = -2; zz <= 2; ++zz)
                    {
                        const auto index_zz = zz + 2;
                        const auto index_z = z + zz;

                        for (int yy = -2; yy <= 2; ++yy)
                        {
                            const auto index_yy = yy + 2;
                            const auto index_y = y + yy;

                            for (int xx = -2; xx <= 2; ++xx)
                            {
                                const auto index_xx = xx + 2;
                                const auto index_x = x + xx;

                                const auto index_block = index_xx + 5 * (index_yy + 5 * index_zz);
                                const auto index_orig = index_x + original_vector_field.dimensions()[0]
                                    * (index_y + original_vector_field.dimensions()[1] * index_z);

                                vector_block->SetTuple(index_block, vector_field_deformed->GetTuple(index_orig));
                            }
                        }
                    }

                    // For each degree of freedom, calculate derivative
                    for (int zz = -2; zz <= 2; ++zz)
                    {
                        const auto index_zz = zz + 2;

                        for (int yy = -2; yy <= 2; ++yy)
                        {
                            const auto index_yy = yy + 2;

                            for (int xx = -2; xx <= 2; ++xx)
                            {
                                const auto index_xx = xx + 2;
                                const auto index_block = index_xx + 5 * (index_yy + 5 * index_zz);

                                // Move node in x- and y-direction respectively (step size?)
                                // Update Jacobians of deformation (necessary? how?)
                                // Calculate difference between original and deformed curvature gradient for central node
                            }
                        }
                    }
                }
            }
        }



        // TODO: Update vector_field_deformed




        // Calculate new gradient difference
        for (vtkIdType i = 0; i < original_curvature.curvature_gradient->GetNumberOfTuples(); ++i)
        {
            gradient_difference->SetValue(i,
                deformed_curvature.curvature_gradient->GetComponent(i, 0) - original_curvature.curvature_gradient->GetComponent(i, 0));
        }

        // Calculate new min, max error
        const auto new_error_min = gradient_difference->GetDataTypeValueMin();
        const auto new_error_max = gradient_difference->GetDataTypeValueMax();

        if (new_error_min > error_min)
        {
            std::cout << "New minimum error increased from " << error_min << " to " << new_error_min << "." << std::endl;
        }
        if (new_error_max > error_max)
        {
            std::cout << "New maximum error increased from " << error_max << " to " << new_error_max << "." << std::endl;
        }

        error_min = new_error_min;
        error_max = new_error_max;

        if (error_max < this->Error)
        {
            converged = true;
        }
    }

    return 1;
}
