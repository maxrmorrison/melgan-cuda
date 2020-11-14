#include <iostream>
#include <string>

#include <boost/program_options.hpp>

#include "melgan.hpp"


/* Run melgan inference */
int main(int argc, char **argv)
{
    namespace po = boost::program_options;

    // Declare command-line arguments
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Produce a help message")
        (
            "mels",
            po::value<std::string>(),
            "Binary file containing 32-bit float mels"
        )
        (
            "frames",
            po::value<unsigned int>(),
            "Number of frames of mels to load"
        )
        (
            "output",
            po::value<std::string>(),
            "Output binary file to store 32-bit float audio"
        );

    // Parse command-line arguments
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Help
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    // Infer
    if (vm.count("mels") && vm.count("frames") && vm.count("output")) {
        infer_from_file_to_file(vm["mels"].as<std::string>(),
                                vm["frames"].as<unsigned int>(),
                                vm["output"].as<std::string>());
    }

    return 0;
}
