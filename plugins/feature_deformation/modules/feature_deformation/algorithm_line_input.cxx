#include "algorithm_line_input.h"

#include "hash.h"

#include "vtkCellArray.h"
#include "vtkIdList.h"
#include "vtkPoints.h"
#include "vtkPolyData.h"
#include "vtkSmartPointer.h"

#include "vtkCellData.h"
#include "vtkPointData.h"

#include "Eigen/Dense"

#include <array>
#include <iostream>

void algorithm_line_input::set_input(vtkPolyData* lines, int selected_line_id)
{
    this->input_lines = lines;
    this->selected_line_id = selected_line_id;
}

std::uint32_t algorithm_line_input::calculate_hash() const
{
    if (this->input_lines == nullptr)
    {
        std::cerr << "No input lines found" << std::endl;

        return -1;
    }

    return jenkins_hash(this->input_lines->GetPoints()->GetMTime(), this->input_lines->GetLines()->GetMTime(), this->selected_line_id);
}

bool algorithm_line_input::run_computation()
{
    // Sanity check
    if (this->selected_line_id < 0 || this->selected_line_id >= this->input_lines->GetLines()->GetNumberOfCells())
    {
        std::cerr << "Line index out of range" << std::endl;
        return false;
    }

    if (!this->is_quiet()) std::cout << "Loading input lines" << std::endl;

    // Get lines
    this->results.lines.resize(this->input_lines->GetNumberOfPoints());

    #pragma omp parallel for
    for (vtkIdType p = 0; p < this->input_lines->GetNumberOfPoints(); ++p)
    {
        std::array<double, 3> point;
        this->input_lines->GetPoints()->GetPoint(p, point.data());

        this->results.lines[p] = { static_cast<float>(point[0]), static_cast<float>(point[1]), static_cast<float>(point[2]) };
    }

    // Extract selected line
    auto point_list = vtkSmartPointer<vtkIdList>::New();
    std::size_t line_index = 0;

    this->input_lines->GetLines()->InitTraversal();
    while (this->input_lines->GetLines()->GetNextCell(point_list))
    {
        if (line_index == this->selected_line_id)
        {
            this->results.selected_line.resize(point_list->GetNumberOfIds());

            #pragma omp parallel for
            for (vtkIdType point_index = 0; point_index < point_list->GetNumberOfIds(); ++point_index)
            {
                std::array<double, 3> point;
                this->input_lines->GetPoints()->GetPoint(point_list->GetId(point_index), point.data());

                this->results.selected_line[point_index] << static_cast<float>(point[0]), static_cast<float>(point[1]), static_cast<float>(point[2]);
            }
        }

        ++line_index;
    }

    if (this->results.selected_line.size() < 3)
    {
        if (!this->is_quiet()) std::cout << "Line consists only of one segment -- nothing to do" << std::endl;

        return false;
    }

    // Set input as output
    this->results.input_lines = this->input_lines;

    return true;
}

void algorithm_line_input::cache_load() const
{
    if (!this->is_quiet()) std::cout << "Loading input lines from cache" << std::endl;
}

const algorithm_line_input::results_t& algorithm_line_input::get_results() const
{
    return this->results;
}

algorithm_input::points_t algorithm_line_input::get_points() const
{
    return algorithm_input::points_t{ this->results.lines, this->get_hash(), this->is_valid() };
}
