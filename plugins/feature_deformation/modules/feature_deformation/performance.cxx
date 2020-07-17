#include "performance.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>

performance::performance(std::ostream* const output_stream) :
    output_stream(output_stream), own(false), active(false), start_time(clock_t::now()), label("")
{
}

performance::performance(const std::string& file_name, int file_options) :
    output_stream(new std::ofstream(file_name, file_options)), own(true), active(false), start_time(clock_t::now()), label("")
{
}

performance::~performance() noexcept
{
    stop();

    if (own && this->output_stream != nullptr)
    {
        delete this->output_stream;
    }
}

void performance::start(std::string label)
{
    this->start_time = clock_t::now();
    this->label = label;

    this->active = true;
}

std::tuple<performance::duration_t, std::string> performance::next_interval(std::string label)
{
    if (!this->active)
    {
        start(label);

        return std::make_tuple(performance::duration_t::zero(), "");
    }

    // Calculate duration
    const auto stop_time = clock_t::now();

    const auto duration = std::chrono::duration_cast<duration_t>(stop_time - this->start_time);

    if (this->output_stream != nullptr)
    {
        (*this->output_stream) << this->label << "," << duration.count() << "," << this->unit() << std::endl;
    }

    // Return message and duration
    std::stringstream ss_1, ss_2, ss_3, ss_4;
    ss_1 << "Performance measure for '" << this->label << "':";

    ss_2.width(55);
    ss_2.fill(' ');
    ss_2 << std::left << ss_1.str();

    ss_3.width(10);
    ss_3.fill(' ');
    ss_3 << std::right << duration.count() << " " << this->unit();

    ss_4 << "\x1B[36m" << ss_2.str() << ss_3.str() << "\033[0m";

    // Start new measurement
    this->start_time = stop_time;
    this->label = label;

    return std::make_tuple(duration, ss_4.str());
}

std::tuple<performance::duration_t, std::string> performance::stop()
{
    if (this->active)
    {
        const auto result = next_interval("");
        (*this->output_stream) << std::endl;

        this->active = false;

        return result;
    }

    return std::make_tuple(performance::duration_t::zero(), "");
}
