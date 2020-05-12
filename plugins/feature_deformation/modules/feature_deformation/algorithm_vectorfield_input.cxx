#include "algorithm_vectorfield_input.h"

#include "algorithm_grid_input.h"
#include "hash.h"

#include "vtkImageData.h"
#include "vtkPointData.h"

#include <memory>

void algorithm_vectorfield_input::set_input(std::shared_ptr<const algorithm_grid_input> grid, const std::string& array_name)
{
    this->input_grid = grid;
    this->array_name = array_name;
}

std::uint32_t algorithm_vectorfield_input::calculate_hash() const
{
    if (!this->input_grid->is_valid() && this->input_grid->get_results().grid->GetPointData()->GetArray(this->array_name.c_str()) != nullptr)
    {
        return -1;
    }

    return jenkins_hash(this->input_grid->get_hash(), this->array_name,
        this->input_grid->get_results().grid->GetPointData()->GetArray(this->array_name.c_str())->GetMTime());
}

bool algorithm_vectorfield_input::run_computation()
{
    std::cout << "Loading input vector field" << std::endl;

    this->results.vector_field = this->input_grid->get_results().grid->GetPointData()->GetArray(this->array_name.c_str());

    if (this->results.vector_field->GetNumberOfComponents() != 3)
    {
        std::cerr << "The data dimension of the vector field has to be 3" << std::endl;
        return false;
    }

    return true;
}

void algorithm_vectorfield_input::cache_load() const
{
    std::cout << "Loading input vector field from cache" << std::endl;
}

const algorithm_vectorfield_input::results_t& algorithm_vectorfield_input::get_results() const
{
    return this->results;
}
