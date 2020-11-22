## Calibration and Documentation

This repository holds diagnostic and research for calibrating option prices using the empirical characteristic function.  The main contributions are:

* The use of the empirical characteristic function for use in parametric option pricing calibration
* The use of genetic and meta-heurstic algorithms in overcoming the highly non-linear optimization problem

However, after several experiments, it appears that using "traditional" L-BFGS directly on the option prices (not transforming into the complex domain) obtains superior results.


## Requirements

The documentation is written in [R Sweave](https://www.r-bloggers.com/getting-started-with-sweave-r-latex-eclipse-statet-texlipse/).  The application is written in [Rust](https://www.rust-lang.org/en-US/).  To efficiently generate the json files needed for the documentation, use [Node](https://nodejs.org/en/). 

## Steps to run

* Clone this repo and cd into the folder
* `cargo build --release`
* `node index `
* Open [OptionCalibration](./docs/OptionCalibration.rnw) in a Sweave/Latex editor (eg RStudio) and compile.

## Relevant links

The difficult work is done by some of my dependent Crates:  
* [fang_oost](https://crates.io/crates/fang_oost)
* [fang_oost_option](https://crates.io/crates/fang_oost_option)
* [cuckoo](https://crates.io/crates/cuckoo)
* [cf_functions](https://crates.io/crates/cf_functions)
