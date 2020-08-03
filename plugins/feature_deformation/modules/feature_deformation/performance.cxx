#include "performance.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

performance::ostream_ptr_t::ostream_ptr_t(std::ostream* pointer)
{
    this->ostream_ptr = pointer;
    this->ostream_smart_ptr = nullptr;
}

performance::ostream_ptr_t::ostream_ptr_t(std::unique_ptr<std::ostream>&& pointer)
{
    this->ostream_ptr = pointer.get();
    this->ostream_smart_ptr = std::move(pointer);
}

performance::ostream_ptr_t::operator std::ostream* () const
{
    return this->ostream_ptr;
}

performance::performance(std::ostream* const output_stream, const style output_style) : active(false), start_time(clock_t::now()), label("")
{
    add_output(output_stream, output_style);
}

performance::performance(const std::string& file_name, int file_options, const style output_style) : active(false), start_time(clock_t::now()), label("")
{
    add_output(file_name, file_options, output_style);
}

performance::~performance() noexcept
{
    stop();
}

void performance::add_output(std::ostream* output_stream, const style output_style)
{
    if (output_stream == nullptr)
    {
        throw std::runtime_error("Performance log output stream must not be null.");
    }

    this->output_streams.push_back(std::make_pair(ostream_ptr_t(output_stream), output_style));
}

void performance::add_output(const std::string& file_name, int file_options, const style output_style)
{
    this->output_streams.push_back(std::make_pair(ostream_ptr_t(std::make_unique<std::ofstream>(file_name, file_options)), output_style));
}

void performance::start(std::string label)
{
    this->start_time = clock_t::now();
    this->label = label;

    this->active = true;
}

void performance::next_interval(std::string label)
{
    if (!this->active)
    {
        start(label);

        return;
    }

    // Calculate duration
    const auto stop_time = clock_t::now();

    const auto duration = std::chrono::duration_cast<duration_t>(stop_time - this->start_time);

    // Create output message
    std::stringstream ss_1, ss_2, ss_3;
    ss_1 << "Performance measure for '" << this->label << "':";

    ss_2.width(55);
    ss_2.fill(' ');
    ss_2 << std::left << ss_1.str();

    ss_3.width(10);
    ss_3.fill(' ');
    ss_3 << std::right << duration.count() << " " << this->unit();

    // Output CSV-style or message
    for (auto& output_stream : this->output_streams)
    {
        if (std::get<0>(output_stream) != nullptr)
        {
            auto& output = *std::get<0>(output_stream);
            const auto output_style = std::get<1>(output_stream);

            if (output_style == style::csv)
            {
                output << this->label << "," << duration.count() << "," << this->unit() << std::endl;
            }
            else if (output_style == style::message)
            {
                output << ss_2.str() << ss_3.str() << std::endl;
            }
            else if (output_style == style::colored_message)
            {
                output << "\x1B[36m" << ss_2.str() << ss_3.str() << "\033[0m" << std::endl;
            }
        }
    }

    // Start new measurement
    this->start_time = stop_time;
    this->label = label;
}

void performance::stop()
{
    if (this->active)
    {
        next_interval("");

        for (auto& output_stream : this->output_streams)
        {
            if (std::get<0>(output_stream) != nullptr)
            {
                auto& output = *std::get<0>(output_stream);
                const auto output_style = std::get<1>(output_stream);

                if (output_style == style::csv)
                {
                    output << std::endl;
                }
            }
        }

        this->active = false;
    }
}
