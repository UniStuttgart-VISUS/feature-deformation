#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

class performance
{
public:
    using clock_t = std::chrono::steady_clock;
    using time_point_t = std::chrono::time_point<clock_t>;
    using duration_t = std::chrono::milliseconds;

    enum class style
    {
        csv, message, colored_message
    };

private:
    static std::string unit() { return "ms"; }

public:
    /// Constructor, receiving an output stream for CSV-style output of measured timings
    explicit performance(std::ostream* output_stream, style output_style);

    /// Constructor, receiving file information for CSV-style output of measured timings
    performance(const std::string& file_name, int file_options, style output_style);

    /// Destructor, stopping the last measurement
    virtual ~performance() noexcept;

    /// Add another output stream for CSV-style output of measured timings
    void add_output(std::ostream* output_stream, style output_style);

    /// Add another file output stream for CSV-style output of measured timings
    void add_output(const std::string& file_name, int file_options, style output_style);

    /// Start measuring, using a label for description
    void start(std::string label);

    /// Start the next interval, outputting the results from the previous interval, if any
    void next_interval(std::string label);

    /// Stop measuring, outputting the results from the previous interval, if any
    void stop();

private:
    /// Pointer to an ostream, which is either owned or not owned by this object
    struct ostream_ptr_t
    {
    public:
        explicit ostream_ptr_t(std::ostream* pointer);
        explicit ostream_ptr_t(std::unique_ptr<std::ostream>&& pointer);

        operator std::ostream* () const;

    private:
        std::ostream* ostream_ptr;
        std::unique_ptr<std::ostream> ostream_smart_ptr;
    };

    /// Output stream for CSV-style data
    std::vector<std::pair<ostream_ptr_t, style>> output_streams;

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
#define __init_perf(output_stream, output_style) performance _std_measure(&output_stream, output_style);
#define __init_perf_file(file_name, file_options, output_style) performance _std_measure(file_name, file_options, output_style);
#define __add_perf(output_stream, output_style) _std_measure.add_output(&output_stream, output_style);
#define __add_perf_file(file_name, file_options, output_style) _std_measure.add_output(file_name, file_options, output_style);
#define __next_perf_measure(label) _std_measure.next_interval(label);
#endif
