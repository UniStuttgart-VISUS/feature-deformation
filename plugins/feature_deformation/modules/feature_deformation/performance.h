#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <tuple>

class performance
{
public:
    using clock_t = std::chrono::steady_clock;
    using time_point_t = std::chrono::time_point<clock_t>;
    using duration_t = std::chrono::milliseconds;

private:
    static std::string unit() { return "ms"; }

public:
    /// Constructor, receiving an output stream for CSV-style output of measured timings
    explicit performance(std::ostream* output_stream = nullptr);

    /// Constructor, receiving file information for CSV-style output of measured timings
    performance(const std::string& file_name, int file_options);

    /// Destructor, stopping the last measurement
    virtual ~performance() noexcept;

    /// Start measuring, using a label for description
    void start(std::string label);

    /// Start the next interval, outputting the results from the previous interval, if any
    std::tuple<duration_t, std::string> next_interval(std::string label);

    /// Stop measuring, outputting the results from the previous interval, if any
    std::tuple<duration_t, std::string> stop();

private:
    /// Output stream for CSV-style data
    std::ostream* const output_stream;

    /// Is the stream created by this class?
    const bool own;

    /// Is the object set to measure?
    bool active;

    /// Start time of the current measure interval
    time_point_t start_time;

    /// Label of the current measure interval
    std::string label;
};

#ifdef __no_performance_measure
#define __initialize_peformance_measure(output_stream)
#define __next_performance_measure(label, quiet)
#else
#define __initialize_peformance_measure(output_stream) performance _std_measure(&output_stream);
#define __initialize_file_peformance_measure(file_name, file_options) performance _std_measure(file_name, file_options);
#define __next_performance_measure(label, quiet) { const auto val = _std_measure.next_interval(label); if (!quiet) std::cout << std::get<1>(val) << std::endl; }
#endif
