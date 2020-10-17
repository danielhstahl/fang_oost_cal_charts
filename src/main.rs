extern crate argmin;
extern crate cf_functions;
extern crate cuckoo;
extern crate fang_oost_option;
extern crate finitediff;
extern crate num_complex;
extern crate optimization;
extern crate rayon;
use argmin::prelude::*;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use finitediff::FiniteDiff;
use rayon::prelude::*;
use std::collections;
#[macro_use]
extern crate serde_json;
#[macro_use]
extern crate serde_derive;

use fang_oost_option::{option_calibration, option_pricing};
use num_complex::Complex;
use optimization::{Func, GradientDescent, Minimizer, NumericalDifferentiation};
use std::f64::consts::PI;
use std::fs::File;
use std::io::prelude::*;

#[derive(Serialize, Deserialize)]
struct OptionRate {
    rate: f64,
    maturity: f64,
    options: Vec<option_calibration::OptionData>,
}
const NUM_SIMS: usize = 1500;
const NEST_SIZE: usize = 25;
const TOL: f64 = 0.000001;
const STRIKE_MULTIPLIER: f64 = 10.0;
const NUM_U: usize = 256;
const NUM_PLOT: usize = 256;
const NUM_INTEGRATE: usize = 4096;
const NUM_DISCRETE_U_DEFAULT: usize = 15;

fn get_u(n: usize) -> Vec<f64> {
    let du = 2.0 * PI / (n as f64);
    (1..n).map(|index| (index as f64) * du).collect()
}

#[derive(Serialize, Deserialize)]
struct SplineResults {
    strike: f64,
    price: f64,
    actual: f64,
}

#[derive(Serialize, Deserialize)]
struct EmpiricalResults {
    strike: f64,
    actual: f64,
}

#[derive(Serialize, Deserialize)]
struct IntegrationResults {
    u: f64,
    estimate_re: f64,
    estimate_im: f64,
    exact_re: f64,
    exact_im: f64,
}

#[derive(Serialize, Deserialize)]
struct ParamaterResults {
    parameter: String,
    actual: f64,
    optimal: f64,
}
fn get_discount(rate: f64, maturity: f64) -> f64 {
    (-rate * maturity).exp()
}

fn print_spline<T>(
    file_name: &str,
    log_cf: T,
    observed_strikes_options: &[option_calibration::OptionData],
    obj_params: &[f64],
    min_strike: f64,
    max_strike: f64,
    max_strike_display: f64,
    rate: f64,
    maturity: f64,
    asset: f64,
) -> std::io::Result<()>
where
    T: Fn(&Complex<f64>, &[f64]) -> Complex<f64>
        + std::marker::Sync
        + std::marker::Sized
        + std::marker::Send,
{
    let discount = get_discount(rate, maturity);

    let s = option_calibration::get_option_spline(
        observed_strikes_options,
        asset,
        discount,
        min_strike,
        max_strike,
    ); //normalizes the strikes and options...maybe I should change this to so that the input to the function returned is non-normalized?

    let max_log_strike = (max_strike_display / asset).ln();
    let log_dk = (2.0 * max_log_strike) / ((NUM_PLOT - 1) as f64);
    let mut dk_array = vec![];
    dk_array
        .extend(&mut (0..NUM_PLOT).map(|index| (max_log_strike - (index as f64) * log_dk).exp()));
    let option_prices_log_dk = option_pricing::fang_oost_call_price(
        NUM_U,
        1.0,
        &dk_array,
        max_strike,
        rate,
        maturity,
        |u| (rate * maturity * u + log_cf(u, obj_params)).exp(),
    ); //this prices options over a larger range of strikes, normalized around asset=1.0

    let json_results_synthetic = json!(option_prices_log_dk
        .iter()
        .zip(dk_array.iter())
        .rev()
        .enumerate()
        //.filter(|(index, _)| index > &0 && index < &max_option_price_index)
        .map(|(_, (option_price, k))| SplineResults {
            strike: k.ln() - rate * maturity,
            price: option_calibration::max_zero_or_number(s(*k)),
            actual: option_price - option_calibration::max_zero_or_number(1.0 - k * discount)
        })
        .collect::<Vec<_>>());
    let json_results_empirical = json!(observed_strikes_options
        .iter()
        .map(
            |option_calibration::OptionData { strike, price, .. }| EmpiricalResults {
                strike: (strike / asset).ln() - rate * maturity,
                actual: price / asset
                    - option_calibration::max_zero_or_number(1.0 - strike * discount / asset)
            }
        )
        .collect::<Vec<_>>());
    let json_results = json!({
        "synthetic":json_results_synthetic,
        "empirical":json_results_empirical
    });
    let mut file = File::create(format!("docs/spline_{}.json", file_name))?;
    file.write_all(json_results.to_string().as_bytes())?;
    Ok(())
}

fn print_estimated_cf<T>(
    file_name: &str,
    log_cf: T,
    obj_params: &[f64],
    u_array: &[f64],
    phis: &[Complex<f64>],
) -> std::io::Result<()>
where
    T: Fn(&Complex<f64>, &[f64]) -> Complex<f64>
        + std::marker::Sync
        + std::marker::Sized
        + std::marker::Send,
{
    let json_results = json!(phis
        .iter()
        .zip(u_array.iter())
        .map(|(phi_estimate, u)| {
            let exact = log_cf(&Complex::new(1.0, *u), &obj_params);
            IntegrationResults {
                u: *u,
                estimate_re: phi_estimate.re,
                estimate_im: phi_estimate.im,
                exact_re: exact.re,
                exact_im: exact.im,
            }
        })
        .collect::<Vec<_>>());
    let mut file = File::create(format!("docs/integrate_{}.json", file_name))?;
    file.write_all(json_results.to_string().as_bytes())?;
    Ok(())
}
fn get_obj_fn<'a, 'b: 'a, T>(
    phi_hat: &'b [Complex<f64>],
    u_array: &'b [f64],
    maturity: f64,
    log_cf_fn: T,
) -> impl Fn(&[f64]) -> f64 + 'a
where
    T: Fn(&Complex<f64>, &[f64]) -> Complex<f64> + 'b,
{
    let local_log_cf = move |u: &Complex<f64>, _t: f64, params: &[f64]| log_cf_fn(u, params);
    move |params| {
        option_calibration::obj_fn_cmpl(&phi_hat, &u_array, &params, maturity, &local_log_cf)
            / (phi_hat.len() as f64)
    }
}

fn get_obj_fn_mse<'a, 'b: 'a, T>(
    option_datum: &'b [option_calibration::OptionDataMaturity],
    num_u: usize,
    asset: f64,
    max_strike: f64,
    rate: f64,
    cf_fn: T,
) -> impl Fn(&[f64]) -> f64 + 'a
where
    T: Fn(&Complex<f64>, f64, &[f64]) -> Complex<f64> + 'b + Sync,
{
    move |params| {
        option_calibration::obj_fn_real(
            num_u,
            asset,
            &option_datum,
            max_strike,
            rate,
            &params,
            &cf_fn,
        )
    }
}

fn print_optimal_parameters(
    file_name: &str,
    log_cf: &dyn Fn(&Complex<f64>, &[f64]) -> Complex<f64>,
    obj_params: &[f64],
    ul: &[cuckoo::UpperLower],
    param_names: &[&str],
    u_array: Vec<f64>,
    phis: Vec<Complex<f64>>,
    maturity: f64,
) -> std::io::Result<()> {
    //let local_cf=|u, t, params|log_cf(u, params);
    let obj_fn = get_obj_fn(&phis, &u_array, maturity, &log_cf);

    let (optimal_parameters, _) = cuckoo::optimize(&obj_fn, ul, NEST_SIZE, NUM_SIMS, TOL, || {
        cuckoo::get_rng_system_seed()
    })
    .unwrap();

    let json_results = json!(param_names
        .iter()
        .zip(obj_params.iter())
        .zip(optimal_parameters.iter())
        .map(|((name, param), optimal)| {
            ParamaterResults {
                parameter: name.to_string(),
                actual: *param,
                optimal: *optimal,
            }
        })
        .collect::<Vec<_>>());
    let mut file = File::create(format!("docs/estimate_{}.json", file_name))?;
    file.write_all(json_results.to_string().as_bytes())?;
    Ok(())
}

fn print_optimal_parameters_sgd(
    file_name: &str,
    log_cf: &dyn Fn(&Complex<f64>, &[f64]) -> Complex<f64>,
    obj_params: &[f64],
    ul: &[cuckoo::UpperLower],
    param_names: &[&str],
    u_array: Vec<f64>,
    phis: Vec<Complex<f64>>,
    maturity: f64,
) -> std::io::Result<()> {
    let obj_fn = get_obj_fn(&phis, &u_array, maturity, &log_cf);

    let function = NumericalDifferentiation::new(Func(|x: &[f64]| obj_fn(x)));

    let minimizer = GradientDescent::new();
    let mid: Vec<f64> = ul
        .iter()
        .map(|cuckoo::UpperLower { upper, lower }| (upper + lower) * 0.5)
        .collect();
    let solution = minimizer.minimize(&function, mid);

    let json_results = json!(param_names
        .iter()
        .zip(obj_params.iter())
        .zip(solution.position.iter())
        .map(|((name, param), optimal)| {
            ParamaterResults {
                parameter: name.to_string(),
                actual: *param,
                optimal: *optimal,
            }
        })
        .collect::<Vec<_>>());
    let mut file = File::create(format!("docs/estimate_{}.json", file_name))?;
    file.write_all(json_results.to_string().as_bytes())?;
    Ok(())
}

struct LogCF<'a> {
    obj_fn: &'a dyn Fn(&[f64]) -> f64,
}

impl ArgminOp for LogCF<'_> {
    type Param = Vec<f64>;
    type Output = f64;
    type Hessian = Vec<f64>;
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, param: &Vec<f64>) -> Result<f64, Error> {
        let ofn = self.obj_fn;
        Ok(ofn(&param))
    }

    fn gradient(&self, param: &Vec<f64>) -> Result<Vec<f64>, Error> {
        let ofn = self.obj_fn;
        Ok((*param).central_diff(&|x| ofn(&x)))
    }
}
fn print_optimal_parameters_lbfgs(
    file_name: &str,
    log_cf: &dyn Fn(&Complex<f64>, &[f64]) -> Complex<f64>,
    obj_params: &[f64],
    ul: &[cuckoo::UpperLower],
    param_names: &[&str],
    u_array: Vec<f64>,
    phis: Vec<Complex<f64>>,
    maturity: f64,
) -> std::io::Result<()> {
    //let local_cf=|u, t, params|log_cf(u, params);
    let obj_fn = get_obj_fn(&phis, &u_array, maturity, &log_cf);

    let obj_fn = LogCF { obj_fn: &obj_fn };

    let linesearch = MoreThuenteLineSearch::new();

    // Set up solver
    // m between 3 and 20 yield "good results" according to
    // http://www.apmath.spbu.ru/cnsa/pdf/monograf/Numerical_Optimization2006.pdf
    let solver = LBFGS::new(linesearch, 7);

    let mid: Vec<f64> = ul
        .iter()
        .map(|cuckoo::UpperLower { upper, lower }| (upper + lower) * 0.5)
        .collect();

    let res = Executor::new(obj_fn, solver, mid)
        .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
        .max_iters(100)
        .run()
        .unwrap();

    let json_results = json!(param_names
        .iter()
        .zip(obj_params.iter())
        .zip(res.state.get_best_param())
        .map(|((name, param), optimal)| {
            ParamaterResults {
                parameter: name.to_string(),
                actual: *param,
                optimal: optimal,
            }
        })
        .collect::<Vec<_>>());
    let mut file = File::create(format!("docs/estimate_{}.json", file_name))?;
    file.write_all(json_results.to_string().as_bytes())?;
    Ok(())
}

fn print_mse_results<T>(
    file_name: &str,
    log_cf: T,
    strikes: &[f64],
    obj_params: &[f64],
    ul: &[cuckoo::UpperLower],
    param_names: &[&str],
    rate: f64,
    maturity: f64,
    asset: f64,
) -> std::io::Result<()>
where
    T: Fn(&Complex<f64>, &[f64]) -> Complex<f64>
        + std::marker::Sync
        + std::marker::Sized
        + std::marker::Send,
{
    if ul.len() != param_names.len() {
        panic!("Requires ul and param_names to be same length");
    }
    if obj_params.len() != param_names.len() {
        panic!("Requires obj_params and param_names to be the same length");
    }

    let max_strike = STRIKE_MULTIPLIER * strikes.last().expect("Requires at least one strike");
    let option_prices = option_pricing::fang_oost_call_price(
        NUM_U,
        asset,
        &strikes,
        max_strike,
        rate,
        maturity,
        |u| (rate * maturity * u + log_cf(u, obj_params)).exp(),
    );
    let observed_strikes_options: Vec<option_calibration::OptionData> = option_prices
        .iter()
        .enumerate()
        .zip(strikes.iter())
        .map(|((_, option), strike)| {
            println!("price: {}", option);
            println!("strike: {}", strike);
            option_calibration::OptionData {
                price: *option,
                strike: *strike,
            }
        })
        .collect();
    let full_data: Vec<option_calibration::OptionDataMaturity> =
        vec![option_calibration::OptionDataMaturity {
            maturity: maturity,
            option_data: observed_strikes_options,
        }];
    let obj_fn = get_obj_fn_mse(
        &full_data,
        NUM_U,
        asset,
        max_strike,
        rate,
        |u, t: f64, params: &[f64]| (rate * t * u + log_cf(u, params)).exp(),
    );

    let obj_fn = LogCF { obj_fn: &obj_fn };

    let linesearch = MoreThuenteLineSearch::new();

    // Set up solver
    // m between 3 and 20 yield "good results" according to
    // http://www.apmath.spbu.ru/cnsa/pdf/monograf/Numerical_Optimization2006.pdf
    let solver = LBFGS::new(linesearch, 7);

    let mid: Vec<f64> = ul
        .iter()
        .map(|cuckoo::UpperLower { upper, lower }| (upper + lower) * 0.5)
        .collect();

    let res = Executor::new(obj_fn, solver, mid)
        .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
        .max_iters(100)
        .run()
        .unwrap();

    let json_results = json!(param_names
        .iter()
        .zip(obj_params.iter())
        .zip(res.state.get_best_param())
        .map(|((name, param), optimal)| {
            ParamaterResults {
                parameter: name.to_string(),
                actual: *param,
                optimal: optimal,
            }
        })
        .collect::<Vec<_>>());
    let mut file = File::create(format!("docs/estimate_{}.json", file_name))?;
    file.write_all(json_results.to_string().as_bytes())?;
    Ok(())
}

fn print_mse_results_exact_parameters<T>(
    file_name: &str,
    log_cf: T,
    strikes: &[f64],
    obj_params: &[f64],
    ul: &[cuckoo::UpperLower],
    param_names: &[&str],
    rate: f64,
    maturity: f64,
    asset: f64,
) -> std::io::Result<()>
where
    T: Fn(&Complex<f64>, &[f64]) -> Complex<f64>
        + std::marker::Sync
        + std::marker::Sized
        + std::marker::Send,
{
    if ul.len() != param_names.len() {
        panic!("Requires ul and param_names to be same length");
    }
    if obj_params.len() != param_names.len() {
        panic!("Requires obj_params and param_names to be the same length");
    }

    let max_strike = STRIKE_MULTIPLIER * strikes.last().expect("Requires at least one strike");

    let option_prices = option_pricing::fang_oost_call_price(
        NUM_U,
        asset,
        &strikes,
        max_strike,
        rate,
        maturity,
        |u| (rate * maturity * u + log_cf(u, obj_params)).exp(),
    );
    //let end_index = option_prices.len() - 1;
    let observed_strikes_options: Vec<option_calibration::OptionData> = option_prices
        .iter()
        .enumerate()
        //.filter(|(index, _)| index > &0 && index < &end_index)
        //.rev()
        .zip(strikes.iter())
        .map(|((_, option), strike)| option_calibration::OptionData {
            price: *option,
            strike: *strike,
        })
        .collect();
    let full_data: Vec<option_calibration::OptionDataMaturity> =
        vec![option_calibration::OptionDataMaturity {
            maturity: maturity,
            option_data: observed_strikes_options,
        }];
    let obj_fn = get_obj_fn_mse(
        &full_data,
        NUM_U,
        asset,
        max_strike,
        rate,
        |u, t: f64, params: &[f64]| (rate * t * u + log_cf(u, params)).exp(),
    );

    let obj_fn = LogCF { obj_fn: &obj_fn };

    let linesearch = MoreThuenteLineSearch::new();

    // Set up solver
    // m between 3 and 20 yield "good results" according to
    // http://www.apmath.spbu.ru/cnsa/pdf/monograf/Numerical_Optimization2006.pdf
    let solver = LBFGS::new(linesearch, 7);

    let res = Executor::new(obj_fn, solver, obj_params.to_vec())
        .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
        .max_iters(100)
        .run()
        .unwrap();

    let json_results = json!(param_names
        .iter()
        .zip(obj_params.iter())
        .zip(res.state.get_best_param())
        .map(|((name, param), optimal)| {
            ParamaterResults {
                parameter: name.to_string(),
                actual: *param,
                optimal: optimal,
            }
        })
        .collect::<Vec<_>>());
    let mut file = File::create(format!("docs/estimate_{}.json", file_name))?;
    file.write_all(json_results.to_string().as_bytes())?;
    Ok(())
}

fn print_mse_results_cuckoo<T>(
    file_name: &str,
    log_cf: T,
    strikes: &[f64],
    obj_params: &[f64],
    ul: &[cuckoo::UpperLower],
    param_names: &[&str],
    rate: f64,
    maturity: f64,
    asset: f64,
) -> std::io::Result<()>
where
    T: Fn(&Complex<f64>, &[f64]) -> Complex<f64>
        + std::marker::Sync
        + std::marker::Sized
        + std::marker::Send,
{
    if ul.len() != param_names.len() {
        panic!("Requires ul and param_names to be same length");
    }
    if obj_params.len() != param_names.len() {
        panic!("Requires obj_params and param_names to be the same length");
    }

    let max_strike = STRIKE_MULTIPLIER * strikes.last().expect("Requires at least one strike");
    //let min_strike = asset / max_strike;

    //let mut k_array = vec![max_strike];
    //k_array.append(&mut strikes.iter().rev().map(|v| *v).collect());
    //k_array.push(min_strike);

    let option_prices = option_pricing::fang_oost_call_price(
        NUM_U,
        asset,
        &strikes,
        max_strike,
        rate,
        maturity,
        |u| (rate * maturity * u + log_cf(u, obj_params)).exp(),
    );
    let observed_strikes_options: Vec<option_calibration::OptionData> = option_prices
        .iter()
        .enumerate()
        //filter(|(index, _)| index > &0 && index < &end_index
        .zip(strikes.iter())
        .map(|((_, option), strike)| option_calibration::OptionData {
            price: *option,
            strike: *strike,
        })
        .collect();
    let full_data: Vec<option_calibration::OptionDataMaturity> =
        vec![option_calibration::OptionDataMaturity {
            maturity: maturity,
            option_data: observed_strikes_options,
        }];
    let obj_fn = get_obj_fn_mse(
        &full_data,
        NUM_U,
        asset,
        max_strike,
        rate,
        |u, t: f64, params: &[f64]| (rate * t * u + log_cf(u, params)).exp(),
    );
    let (optimal_parameters, _) = cuckoo::optimize(&obj_fn, ul, NEST_SIZE, NUM_SIMS, TOL, || {
        cuckoo::get_rng_system_seed()
    })
    .unwrap();
    let json_results = json!(param_names
        .iter()
        .zip(obj_params.iter())
        .zip(optimal_parameters)
        .map(|((name, param), optimal)| {
            ParamaterResults {
                parameter: name.to_string(),
                actual: *param,
                optimal: optimal,
            }
        })
        .collect::<Vec<_>>());

    let mut file = File::create(format!("docs/estimate_{}.json", file_name))?;
    file.write_all(json_results.to_string().as_bytes())?;
    Ok(())
}
fn print_results<T, U>(
    file_name: &str,
    log_cf: T,
    strikes: &[f64],
    obj_params: &[f64],
    ul: &[cuckoo::UpperLower],
    param_names: &[&str],
    rate: f64,
    maturity: f64,
    asset: f64,
    u_array: Vec<f64>,
    print_optimal_parameters_fn: U,
) -> std::io::Result<()>
where
    T: Fn(&Complex<f64>, &[f64]) -> Complex<f64>
        + std::marker::Sync
        + std::marker::Sized
        + std::marker::Send,
    U: Fn(
        &str,
        &dyn Fn(&Complex<f64>, &[f64]) -> Complex<f64>,
        &[f64],
        &[cuckoo::UpperLower],
        &[&str],
        Vec<f64>,
        Vec<Complex<f64>>,
        f64,
    ) -> std::io::Result<()>,
{
    if ul.len() != param_names.len() {
        panic!("Requires ul and param_names to be same length");
    }
    if obj_params.len() != param_names.len() {
        panic!("Requires obj_params and param_names to be the same length");
    }

    let max_strike = STRIKE_MULTIPLIER * strikes.last().expect("Requires at least one strike");
    let max_strike_display = max_strike * 0.3;
    let min_strike = asset / max_strike;

    /*let mut k_array = vec![max_strike];
    k_array.append(&mut strikes.iter().rev().map(|v| *v).collect());
    k_array.push(min_strike);*/
    let option_prices = option_pricing::fang_oost_call_price(
        NUM_U,
        asset,
        &strikes,
        max_strike,
        rate,
        maturity,
        |u| (rate * maturity * u + log_cf(u, obj_params)).exp(),
    );
    let observed_strikes_options: Vec<option_calibration::OptionData> = option_prices
        .iter()
        .enumerate()
        //.filter(|(index, _)| index > &0 && index < &end_index)
        //.rev()
        .zip(strikes.iter())
        .map(|((_, option), strike)| option_calibration::OptionData {
            price: *option,
            strike: *strike,
        })
        .collect();

    print_spline(
        file_name,
        &log_cf,
        &observed_strikes_options,
        obj_params,
        min_strike,
        max_strike,
        max_strike_display,
        rate,
        maturity,
        asset,
    )?;
    let phis: Vec<Complex<f64>> = option_calibration::generate_fo_estimate(
        &observed_strikes_options,
        &u_array,
        NUM_INTEGRATE,
        asset,
        rate,
        maturity,
        min_strike,
        max_strike,
    )
    .collect();
    //let phis=estimate_of_phi(NUM_INTEGRATE, &u_array);

    print_estimated_cf(file_name, &log_cf, &obj_params, &u_array, &phis)?;

    print_optimal_parameters_fn(
        file_name,
        &log_cf,
        obj_params,
        ul,
        param_names,
        u_array,
        phis,
        maturity,
    )?;
    Ok(())
}

fn print_results_default_u<T>(
    file_name: &str,
    log_cf: T,
    strikes: &[f64],
    obj_params: &[f64],
    ul: &[cuckoo::UpperLower],
    param_names: &[&str],
    rate: f64,
    maturity: f64,
    asset: f64,
) -> std::io::Result<()>
where
    T: Fn(&Complex<f64>, &[f64]) -> Complex<f64>
        + std::marker::Sync
        + std::marker::Sized
        + std::marker::Send,
{
    print_results(
        file_name,
        log_cf,
        strikes,
        obj_params,
        ul,
        param_names,
        rate,
        maturity,
        asset,
        get_u(NUM_DISCRETE_U_DEFAULT),
        print_optimal_parameters,
    )
}

fn print_results_default_u_sgd<T>(
    file_name: &str,
    log_cf: T,
    strikes: &[f64],
    obj_params: &[f64],
    ul: &[cuckoo::UpperLower],
    param_names: &[&str],
    rate: f64,
    maturity: f64,
    asset: f64,
) -> std::io::Result<()>
where
    T: Fn(&Complex<f64>, &[f64]) -> Complex<f64>
        + std::marker::Sync
        + std::marker::Sized
        + std::marker::Send,
{
    print_results(
        file_name,
        log_cf,
        strikes,
        obj_params,
        ul,
        param_names,
        rate,
        maturity,
        asset,
        get_u(NUM_DISCRETE_U_DEFAULT),
        print_optimal_parameters_sgd,
    )
}
fn print_results_default_u_lbfgs<T>(
    file_name: &str,
    log_cf: T,
    strikes: &[f64],
    obj_params: &[f64],
    ul: &[cuckoo::UpperLower],
    param_names: &[&str],
    rate: f64,
    maturity: f64,
    asset: f64,
) -> std::io::Result<()>
where
    T: Fn(&Complex<f64>, &[f64]) -> Complex<f64>
        + std::marker::Sync
        + std::marker::Sized
        + std::marker::Send,
{
    print_results(
        file_name,
        log_cf,
        strikes,
        obj_params,
        ul,
        param_names,
        rate,
        maturity,
        asset,
        get_u(NUM_DISCRETE_U_DEFAULT),
        print_optimal_parameters_lbfgs,
    )
}

#[derive(Serialize, Deserialize)]
struct CalibrationParameters {
    options_and_rate: Vec<OptionRate>,
    asset: f64,
    constraints: collections::HashMap<String, cuckoo::UpperLower>,
}
const STRIKE_RATIO: f64 = 10.0;
fn generate_const_parameters(
    strikes_and_option_prices: &[option_calibration::OptionData],
    asset: f64,
) -> (usize, f64, f64) {
    let n = 1024;
    let option_calibration::OptionData {
        strike: strike_last,
        ..
    } = strikes_and_option_prices
        .last()
        .expect("require at least one strike");
    let max_strike = strike_last * STRIKE_RATIO;
    // reciprocal of max strike, but multiplied
    // by asset to ensure that the range stays
    // appropriate regardless of the asset size.
    // Note that this implies we have to "undo"
    // this later if we want symmetry
    let min_strike = asset / max_strike;
    (n, min_strike, max_strike)
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let print_choice: i32 = args[1].parse().unwrap();
    match print_choice {
        0 => {
            let stock = 10.0;
            let rate = 0.05;
            let sigma = 0.3;
            let maturity = 1.0;
            let constraints = vec![cuckoo::UpperLower {
                lower: 0.0,
                upper: 0.6,
            }];
            let param_names = vec!["sigma"];
            let strikes = vec![
                4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 20.0,
            ];
            let cf = |u: &Complex<f64>, obj_params: &[f64]| {
                let sigma = obj_params[0];
                (-sigma.powi(2) * u * 0.5 + sigma.powi(2) * u * u * 0.5) * maturity
            };
            let obj_params = vec![sigma];
            let u_array = vec![
                -20.0, -15.0, -10.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 10.0, 15.0, 20.0,
            ];
            print_results(
                "black_scholes_test_u",
                cf,
                &strikes,
                &obj_params,
                &constraints,
                &param_names,
                rate,
                maturity,
                stock,
                u_array,
                print_optimal_parameters,
            )
        }
        1 => {
            let stock = 10.0;
            let rate = 0.05;
            let sigma = 0.3;
            let maturity = 0.5;
            let constraints = vec![cuckoo::UpperLower {
                lower: 0.0,
                upper: 0.6,
            }];
            let param_names = vec!["sigma"];
            let strikes = vec![
                4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 20.0,
            ];
            let cf = |u: &Complex<f64>, obj_params: &[f64]| {
                let sigma = obj_params[0];
                (-sigma.powi(2) * u * 0.5 + sigma.powi(2) * u * u * 0.5) * maturity
            };
            let obj_params = vec![sigma];
            print_results_default_u(
                "black_scholes",
                cf,
                &strikes,
                &obj_params,
                &constraints,
                &param_names,
                rate,
                maturity,
                stock,
            )
        }
        2 => {
            let stock = 178.46;
            let rate = 0.0;
            let maturity = 1.0;
            let b: f64 = 0.0398;
            let a = 1.5768;
            let c = 0.5751;
            let rho = -0.5711;
            let v0 = 0.0175;
            let sig = b.sqrt();
            let speed = a;
            let v0_hat = v0 / b;
            let ada_v = c / sig;
            let obj_params = vec![sig, speed, ada_v, rho, v0_hat];

            let constraints = vec![
                cuckoo::UpperLower {
                    lower: 0.0,
                    upper: 0.6,
                },
                cuckoo::UpperLower {
                    lower: 0.0,
                    upper: 2.0,
                },
                cuckoo::UpperLower {
                    lower: 0.0,
                    upper: 4.0,
                },
                cuckoo::UpperLower {
                    lower: -1.0,
                    upper: 1.0,
                },
                cuckoo::UpperLower {
                    lower: 0.0,
                    upper: 2.0,
                },
            ];
            let param_names = vec!["sigma", "speed", "ada_v", "rho", "v0_hat"];
            let strikes = vec![
                95.0, 100.0, 130.0, 150.0, 160.0, 165.0, 170.0, 175.0, 185.0, 190.0, 195.0, 200.0,
                210.0, 240.0, 250.0,
            ];
            let cf = |u: &Complex<f64>, obj_params: &[f64]| {
                let sig = obj_params[0];
                let speed = obj_params[1];
                let ada_v = obj_params[2];
                let rho = obj_params[3];
                let v0_hat = obj_params[4];
                cf_functions::cir_log_mgf_cmp(
                    &(-cf_functions::merton_log_risk_neutral_cf(u, 0.0, 0.0, 0.0, 0.0, sig)),
                    speed,
                    &(speed - ada_v * rho * u * sig),
                    ada_v,
                    maturity,
                    v0_hat,
                )
            };
            print_results_default_u(
                "heston",
                cf,
                &strikes,
                &obj_params,
                &constraints,
                &param_names,
                rate,
                maturity,
                stock,
            )
        }
        3 => {
            let stock = 178.46;
            let rate = 0.0;
            let maturity = 1.0;
            let b: f64 = 0.0398;
            let a = 1.5768;
            let c = 0.5751;
            let rho = -0.5711;
            let v0 = 0.0175;
            let sig = b.sqrt();
            let speed = a;
            let v0_hat = v0 / b;
            let ada_v = c / sig;
            let obj_params = vec![sig, speed, ada_v, rho, v0_hat];

            let constraints = vec![
                cuckoo::UpperLower {
                    lower: 0.0,
                    upper: 0.6,
                },
                cuckoo::UpperLower {
                    lower: 0.0,
                    upper: 2.0,
                },
                cuckoo::UpperLower {
                    lower: 0.0,
                    upper: 4.0,
                },
                cuckoo::UpperLower {
                    lower: -1.0,
                    upper: 1.0,
                },
                cuckoo::UpperLower {
                    lower: 0.0,
                    upper: 2.0,
                },
            ];
            let param_names = vec!["sigma", "speed", "ada_v", "rho", "v0_hat"];
            let strikes = vec![
                95.0, 100.0, 130.0, 150.0, 160.0, 165.0, 170.0, 175.0, 185.0, 190.0, 195.0, 200.0,
                210.0, 240.0, 250.0,
            ];
            let log_cf = |u: &Complex<f64>, obj_params: &[f64]| {
                let sig = obj_params[0];
                let speed = obj_params[1];
                let ada_v = obj_params[2];
                let rho = obj_params[3];
                let v0_hat = obj_params[4];
                cf_functions::cir_log_mgf_cmp(
                    &(-cf_functions::merton_log_risk_neutral_cf(u, 0.0, 0.0, 0.0, 0.0, sig)),
                    speed,
                    &(speed - ada_v * rho * u * sig),
                    ada_v,
                    maturity,
                    v0_hat,
                )
            };
            print_results_default_u_sgd(
                "heston_sgd",
                log_cf,
                &strikes,
                &obj_params,
                &constraints,
                &param_names,
                rate,
                maturity,
                stock,
            )
        }
        4 => {
            let stock = 178.46;
            let rate = 0.0;
            let maturity = 1.0;
            let b: f64 = 0.0398;
            let a = 1.5768;
            let c = 0.5751;
            let rho = -0.5711;
            let v0 = 0.0175;
            let sig = b.sqrt();
            let speed = a;
            let v0_hat = v0 / b;
            let ada_v = c / sig;
            let obj_params = vec![sig, speed, ada_v, rho, v0_hat];

            let constraints = vec![
                cuckoo::UpperLower {
                    lower: 0.0,
                    upper: 0.6,
                },
                cuckoo::UpperLower {
                    lower: 0.0,
                    upper: 2.0,
                },
                cuckoo::UpperLower {
                    lower: 0.0,
                    upper: 4.0,
                },
                cuckoo::UpperLower {
                    lower: -1.0,
                    upper: 1.0,
                },
                cuckoo::UpperLower {
                    lower: 0.0,
                    upper: 2.0,
                },
            ];
            let param_names = vec!["sigma", "speed", "ada_v", "rho", "v0_hat"];
            let strikes = vec![
                95.0, 100.0, 130.0, 150.0, 160.0, 165.0, 170.0, 175.0, 185.0, 190.0, 195.0, 200.0,
                210.0, 240.0, 250.0,
            ];
            let log_cf = |u: &Complex<f64>, obj_params: &[f64]| {
                let sig = obj_params[0];
                let speed = obj_params[1];
                let ada_v = obj_params[2];
                let rho = obj_params[3];
                let v0_hat = obj_params[4];
                cf_functions::cir_log_mgf_cmp(
                    &(-cf_functions::merton_log_risk_neutral_cf(u, 0.0, 0.0, 0.0, 0.0, sig)),
                    speed,
                    &(speed - ada_v * rho * u * sig),
                    ada_v,
                    maturity,
                    v0_hat,
                )
            };
            print_results_default_u_lbfgs(
                "heston_lgbfs",
                log_cf,
                &strikes,
                &obj_params,
                &constraints,
                &param_names,
                rate,
                maturity,
                stock,
            )
        }
        5 => {
            let stock = 178.46;
            let rate = 0.0;
            let maturity = 1.0;
            let b: f64 = 0.0398;
            let a = 1.5768;
            let c = 0.5751;
            let rho = -0.5711;
            let v0 = 0.0175;
            let sig = b.sqrt();
            let speed = a;
            let v0_hat = v0 / b;
            let ada_v = c / sig;
            let obj_params = vec![sig, speed, ada_v, rho, v0_hat];

            let constraints = vec![
                cuckoo::UpperLower {
                    lower: 0.0,
                    upper: 0.6,
                },
                cuckoo::UpperLower {
                    lower: 0.0,
                    upper: 2.0,
                },
                cuckoo::UpperLower {
                    lower: 0.0,
                    upper: 4.0,
                },
                cuckoo::UpperLower {
                    lower: -1.0,
                    upper: 1.0,
                },
                cuckoo::UpperLower {
                    lower: 0.0,
                    upper: 2.0,
                },
            ];
            let param_names = vec!["sigma", "speed", "ada_v", "rho", "v0_hat"];
            let strikes = vec![
                95.0, 100.0, 130.0, 150.0, 160.0, 165.0, 170.0, 175.0, 185.0, 190.0, 195.0, 200.0,
                210.0, 240.0, 250.0,
            ];
            let log_cf = |u: &Complex<f64>, obj_params: &[f64]| {
                let sig = obj_params[0];
                let speed = obj_params[1];
                let ada_v = obj_params[2];
                let rho = obj_params[3];
                let v0_hat = obj_params[4];
                cf_functions::cir_log_mgf_cmp(
                    &(-cf_functions::merton_log_risk_neutral_cf(u, 0.0, 0.0, 0.0, 0.0, sig)),
                    speed,
                    &(speed - ada_v * rho * u * sig),
                    ada_v,
                    maturity,
                    v0_hat,
                )
            };
            print_mse_results_exact_parameters(
                "heston_lgbfs_full_price_exact",
                log_cf,
                &strikes,
                &obj_params,
                &constraints,
                &param_names,
                rate,
                maturity,
                stock,
            )
        }
        6 => {
            let stock = 178.46;
            let rate = 0.0;
            let maturity = 1.0;
            let b: f64 = 0.0398;
            let a = 1.5768;
            let c = 0.5751;
            let rho = -0.5711;
            let v0 = 0.0175;
            let sig = b.sqrt();
            let speed = a;
            let v0_hat = v0 / b;
            let ada_v = c / sig;
            let obj_params = vec![sig, speed, ada_v, rho, v0_hat];

            let constraints = vec![
                cuckoo::UpperLower {
                    lower: 0.0,
                    upper: 0.6,
                },
                cuckoo::UpperLower {
                    lower: 0.0,
                    upper: 2.0,
                },
                cuckoo::UpperLower {
                    lower: 0.0,
                    upper: 4.0,
                },
                cuckoo::UpperLower {
                    lower: -1.0,
                    upper: 1.0,
                },
                cuckoo::UpperLower {
                    lower: 0.0,
                    upper: 2.0,
                },
            ];
            let param_names = vec!["sigma", "speed", "ada_v", "rho", "v0_hat"];
            let strikes = vec![
                95.0, 100.0, 130.0, 150.0, 160.0, 165.0, 170.0, 175.0, 185.0, 190.0, 195.0, 200.0,
                210.0, 240.0, 250.0,
            ];
            let log_cf = |u: &Complex<f64>, obj_params: &[f64]| {
                let sig = obj_params[0];
                let speed = obj_params[1];
                let ada_v = obj_params[2];
                let rho = obj_params[3];
                let v0_hat = obj_params[4];
                cf_functions::cir_log_mgf_cmp(
                    &(-cf_functions::merton_log_risk_neutral_cf(u, 0.0, 0.0, 0.0, 0.0, sig)),
                    speed,
                    &(speed - ada_v * rho * u * sig),
                    ada_v,
                    maturity,
                    v0_hat,
                )
            };
            print_mse_results(
                "heston_lgbfs_full_price",
                log_cf,
                &strikes,
                &obj_params,
                &constraints,
                &param_names,
                rate,
                maturity,
                stock,
            )
        }
        7 => {
            let stock = 178.46;
            let rate = 0.0;
            let maturity = 1.0;
            let b: f64 = 0.0398;
            let a = 1.5768;
            let c = 0.5751;
            let rho = -0.5711;
            let v0 = 0.0175;
            let sig = b.sqrt();
            let speed = a;
            let v0_hat = v0 / b;
            let ada_v = c / sig;
            let obj_params = vec![sig, speed, ada_v, rho, v0_hat];

            let constraints = vec![
                cuckoo::UpperLower {
                    lower: 0.0,
                    upper: 0.6,
                },
                cuckoo::UpperLower {
                    lower: 0.0,
                    upper: 2.0,
                },
                cuckoo::UpperLower {
                    lower: 0.0,
                    upper: 4.0,
                },
                cuckoo::UpperLower {
                    lower: -1.0,
                    upper: 1.0,
                },
                cuckoo::UpperLower {
                    lower: 0.0,
                    upper: 2.0,
                },
            ];
            let param_names = vec!["sigma", "speed", "ada_v", "rho", "v0_hat"];
            let strikes = vec![
                95.0, 100.0, 130.0, 150.0, 160.0, 165.0, 170.0, 175.0, 185.0, 190.0, 195.0, 200.0,
                210.0, 240.0, 250.0,
            ];
            let log_cf = |u: &Complex<f64>, obj_params: &[f64]| {
                let sig = obj_params[0];
                let speed = obj_params[1];
                let ada_v = obj_params[2];
                let rho = obj_params[3];
                let v0_hat = obj_params[4];
                cf_functions::cir_log_mgf_cmp(
                    &(-cf_functions::merton_log_risk_neutral_cf(u, 0.0, 0.0, 0.0, 0.0, sig)),
                    speed,
                    &(speed - ada_v * rho * u * sig),
                    ada_v,
                    maturity,
                    v0_hat,
                )
            };
            print_mse_results_cuckoo(
                "heston_cuckoo_full_price",
                log_cf,
                &strikes,
                &obj_params,
                &constraints,
                &param_names,
                rate,
                maturity,
                stock,
            )
        }

        8 => {
            let cp: CalibrationParameters = serde_json::from_str(&args[2])?;
            cp.options_and_rate.iter().for_each(
                |OptionRate {
                     maturity,
                     rate,
                     options,
                 }| {
                    let (_n, min_strike, max_strike) =
                        generate_const_parameters(&options, cp.asset);
                    let discount = (-rate * maturity).exp();
                    let s = option_calibration::get_option_spline(
                        &options, cp.asset, discount, min_strike, max_strike,
                    ); //normalizes the strikes and options...maybe I should change this to so that the input to the function returned is non-normalized?
                    let max_log_strike = (max_strike * 0.3 / cp.asset).ln();
                    let log_dk = (2.0 * max_log_strike) / ((NUM_PLOT - 1) as f64);
                    let mut dk_array = vec![];
                    dk_array.extend(
                        &mut (0..NUM_PLOT)
                            .map(|index| (max_log_strike - (index as f64) * log_dk).exp()),
                    );
                    let json_results_synthetic = json!(dk_array
                        .iter()
                        .rev()
                        .enumerate()
                        .map(|(_, k)| EmpiricalResults {
                            strike: k.ln() - rate * maturity,
                            actual: option_calibration::max_zero_or_number(s(*k))
                        })
                        .collect::<Vec<_>>());
                    let mut file =
                        File::create(format!("docs/spline_{:.*}.json", 3, maturity)).unwrap();

                    match file.write_all(json_results_synthetic.to_string().as_bytes()) {
                        Ok(_v) => println!("{}", "file written successfully"),
                        Err(e) => println!("{}", format!("{}", e)),
                    }
                    //Ok(())
                },
            );
            Ok(())
        }
        _ => {
            println!("Choice is not valid!");
            Ok(())
        }
    }
}
