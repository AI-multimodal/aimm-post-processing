# import general data science modules
from collections import Counter, namedtuple
from scipy import interpolate
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# import python modules
from collections import namedtuple
Result = namedtuple("Result", "x")


def subtract_background(data, grid, fit_range=(7600, 7700), method='linear'):
    '''
    Subtract background from data.

    Fit the pre-edge data to a line with slope, and subtract slope info from data.

    Parameters
    ----------
    data : array_like
        2-D spectra whose backgroud will be romoved. Each row represents a spectrum,
        each column represents measurement on energy grids.
    grid: array_like
        Shape is (N_features,) corresponding feature grid for the data.
    fit_range : tuple
        The left and right boundary of the data range of which data is fitted.

    Returns
    -------
    data_subtrated : ndarray
        The data of same shape but with slope information subtracted. Must be of the same shape as data.

    Examples
    --------

    '''
    linear_regression = LinearRegression()

    energy_select = (grid >= fit_range[0]) & (grid <= fit_range[1])
    if energy_select.sum() == 0:
        raise ValueError(f"Fit energy out of range, please choose between {grid[0]} and {grid[-1]}")
    energy_to_fit = grid[energy_select].reshape(-1, 1)
    feature_grid = grid.reshape(-1, 1)

    data_to_fit = data[:, energy_select]
    data_subtrated = np.zeros_like(data)
    for i, d in enumerate(data_to_fit):
        if not np.isnan(d).any():
            reg = linear_regression.fit(energy_to_fit, d)
            background = reg.predict(feature_grid)
        else:
            background = 0 # for data contains NaN, set background 0
        data_subtrated[i] = data[i] - background

    return data_subtrated


def normalize(spec, grid=None, mean_range=None, keys=None,
              use_std=True, use_derivative=False):
    '''
    Normalize data: x -> (x-m)/v where m is the mean along axis=1, v is the std along axis=1.
    
    Parameters
    ----------
    data : array_like
        Input data of shape (N_samples,N_features)
    grid : array_like 
        Energy grid of shape(N_features,)
    mean_range : array_like
        list of value [x,y] with x/y being the left/right boundary of the range
    use_std : boolean, whether to use standar
    
    Returns
    -------
    The normalized data with NaN excluded.
    '''

    # calculate mean value for a given energy range
    if mean_range is None:
        mean_select = np.ones(spec.shape[1]).astype('bool')
    else:
        mean_select = (grid > mean_range[0]) & (grid < mean_range[1])
    means_spec = spec[:, mean_select].mean(axis=1).reshape(-1, 1)

    # calculate standard deviation for the whole range
    if use_std:
        stds_spec = spec.std(axis=1).reshape(-1, 1)
    else:  # or not scale to deviation instead
        stds_spec = np.ones_like(spec)

    # calculate the standard deviation for the derivative of the data
    if use_derivative:
        stds_spec_p = derivative(spec[:, mean_select]).std(axis=1).reshape(-1, 1)
    else:
        stds_spec_p = np.ones_like(spec)

    # normalize
    norm_spec = (spec - means_spec) / stds_spec * stds_spec_p


    return norm_spec


def featurize(spec, from_grid, to_grid=None, kind = 'linear', fill_value = 'extrapolate'):
    '''
    Map spectra from original energy grid onto a chosen grid.

    Parameters
    ----------
    spec : array_like
        A 1-D array containing multiple variables and observations. Each element of 
        `spec` represents intensity at the corresponding energy given by `from_grid`.
    from_grid : array_like
        A 1-D array containing the original energy grid on which `spec` value was taken.
        The length of `from_grid` must be the same as the length of `spec` along `axis=1`.
    kind : string
        see `scipy.interpolate.interp1d` for details.
    fill_value : string
        see `scipy.interpolate.interp1d` for details.

    Returns
    -------
    features : array_like
        The spectra intensity mapped on new grid. It must have the same shape of (M, N)
        where M is the number of rows of `spec`, and N is the length of `to_grid`.
        
    '''
    
    if fill_value == 'both_ends':
        fill_value = (spec[0], spec[-1])
    
    try:
        # create a spline interpolation of the origional data
        f = interpolate.interp1d(from_grid, spec, kind=kind, 
                                 fill_value=fill_value, bounds_error=False)
        # collect spec intensities that fall in energy_grid as output.
        features = f(to_grid)
    except ValueError:
        raise
    
    return features



def derivative(data, is_grid = False, stride = 1):
    '''
    Take the "derivative" of the data, i.e. the difference between neighboring points.
    
    Parameters:
    ----------
    data: array of shape(N_samples,N_features)
    stride: take the subtraction of i+stride and i.
    is_grid: if the input is grid (1-D array)
    Return:
    ------
    data_p: "derivative" of the data, of shape (N_samples,N_features-1)
    '''
    step = stride
    if is_grid:
        assert len(data.shape) == 1
        return data[step::step]
    else:
        assert len(data.shape) == 2
    
        data_p = data[:, stride::stride] - data[:, :-stride:stride]
        return data_p

    
def spec2image(spec, grid, image_size=(128,128), energy=(None,None),intensity=(None,None),
               point_density=True):
    '''
    Return the image representation of a 1-D XAS spectrum.
    Paramters:
    ----------
    spec, grid: the 1-D univariate spec data and the grid it is defined on.
    image_size: (float,float) the size of the output image.
    energy: (float,float) the lower and upper limit of the energy range.
    intensity: (float, float) the lower and upper limit of the intensity range.
    point_density: bool. If true the point density of each pixel is returned.
    
    Return:
    -------
    image: the image representation
    '''
    assert None not in intensity, "Error: intensity limit must be specified!"
    # 129 points in intensity grid but 128 seperations
    intensity_grid_2d = np.linspace(intensity[0],intensity[1],image_size[0]+1) 
    step = intensity_grid_2d[1]-intensity_grid_2d[0]
    
    e_lo, e_hi = energy
    if e_lo==None: e_lo = np.min(grid)
    if e_hi==None: e_hi = np.max(grid)
     # 129 points in energy grid but 128 seperations
    energy_grid_2d = np.linspace(e_lo,e_hi,image_size[1]+1)
    image = np.zeros(image_size)
    for i in range(image_size[1]): # point0~point127
        left, right = energy_grid_2d[i:i+2]
        intensity_selected = spec[(grid>=left)&(grid<right)]
        if not len(intensity_selected): 
            continue # no intensity selected
        for y in intensity_selected:
            one_hot = ((intensity_grid_2d-y)>0)&((intensity_grid_2d-y)<=step)
            image[:,i] += one_hot[-1:0:-1].astype(float)
    if not point_density:
        image = (image>0).astype(float)
    
    return image



def find_edge(spec, grid, s=1, k=5, initial_guess=4980, method='trust-krylov',
              output_grid = None, height=0.1, denoise=False,main_peak_range=(4980,5000)):
    
    '''
    Find the edge of an XAS data by finding the peak position of its derivative.
    
    Return:
    -------
    The optimized result that contains the peak position information
    The analytical spec_function and target_function.
    '''
    global Result
    
#     # denoise the data prior initial_guess to avoid spike peak before main peak
    
#     spec[pre_select] = simple_moving_average(spec[pre_select].reshape(1,-1),window=10).flatten()

    
    
    # test if a finer grid would work better.
    if output_grid is not None:
        interp_func = interpolate.interp1d(grid,spec,fill_value='extraploate')
        spec = interp_func(output_grid)
    else:
        output_grid = grid

#     select = (grid<main_peak_range[0]) | (grid>main_peak_range[1])
#     spec = spec[select]
#     output_grid = output_grid[select]
    
    # function for smoothed data and its derivative
    spec_func = UnivariateSpline(output_grid, spec, s=s, k=k)
    target_func = spec_func.derivative()
    
    # jacobian and hessian function
    jac_func = lambda x: -target_func.derivative(n=1)(x)
    hess_func = target_func.derivative(n=2)
    
    # determine the initial guess value
    if initial_guess == 'max': # position of maximum peak
        initial_guess = output_grid[target_func(output_grid).argmax()]
    elif initial_guess == 'first': # position of first peak
        peak, _ = find_peaks(target_func(output_grid), height=height)
        initial_guess = output_grid[peak[0]]
    
    select = (output_grid>main_peak_range[0]) & (output_grid<main_peak_range[1])
    grid2min = output_grid[select]
    if method == "simple":
        peak_pos = np.array([grid2min[np.argmax(target_func(grid2min))]])
        optimize_result = Result(peak_pos)
    else:
        # minimization of the negative of the derivative
        optimize_result = minimize(lambda x: -target_func(x),initial_guess,
                               jac=jac_func, hess=hess_func, method=method)
    
    # return the optimize_result, smoothed spec function and derivative function
    return optimize_result, spec_func, target_func



def shift_edge(spec, grid, shift, output_grid=None):

    '''
    Given a spectrum on a defined grid, and the amount that needs to be shifted.
    Return the shifted spectrum.
    '''
    
    func = interpolate.interp1d(grid,spec,fill_value='extrapolate')
#     finer_grid = np.linspace(grid[0],grid[-1],len(grid)*4-1)
    spec_shift = func(grid+shift)
    
    return spec_shift

def simple_moving_average(spec, window=5):
    '''
    Return the simple moving average of spectra with a rolling window.
    
    Parameters
    ----------
    spec: 2D-array
        The input spectra of shape (N,M) where N and M are the length of data and grid.
    window: int
        The rolling window over which the data is averaged.

    Returns
    -------
    spec_smoothed: 2D-array
        The averaged spectra.
    '''
    left = -np.ceil(-window/2)
    right = np.ceil(window/2)
    spec_smoothed = spec.copy()
    for i in range(spec.shape[1]):
        if i>=left:
            spec_smoothed[:,i] = spec[:,int(i-left):int(i+right)].mean(axis=1)
        else:
            spec_smoothed[:,i] = spec[:,i]
    return spec_smoothed

def df_for_violin(x1, x2, method1="A", method2="B",
                  x_label="name", y_label="value"):
    '''
    Prepare the dataframe for a violin plot. 
    The last 
    '''

    # concatenate both dataframe
    df_list = []
    for x, method in [[x1, method1],[x2,method2]]:
        try:
            df = x.to_frame(name=y_label).assign(method=method,name=x_label)
        except AttributeError:
            df = pd.DataFrame(x,columns=[y_label]).assign(method=method,name=x_label)
        df_list.append(df)
    df = pd.concat(df_list,axis=0)
    return df


def merge_duplicate_energy(grid, spec):
    '''
    Merge multiple data points measrued at the same energy into their average.

    Parameters
    ----------
    grid : array_like
        Any 1-D array that represent energy grid the spectra are measured on.
    spec : array_like
        1-D array of spectra measured on `grid`, must have same length as `grid`.
    
    Return
    ------
    reduced : tuple
        A tuple containing reduced grid and spec.
    '''

    # make sure the energy grid is monotonic
    sorted_indices = np.argsort(grid, axis=0)
    grid = grid[sorted_indices]
    spec = spec[sorted_indices]
    
    # If True then the corresponding selection is a duplicate, append False to make shape consistent.
    select_duplicate = np.append((grid[1:] == grid[:-1]), False)
    energies_duplicate = np.unique(grid[np.where(select_duplicate)[0]])
    
    if len(energies_duplicate) == 0:
        return (grid, spec)
    else:
        for energy in energies_duplicate:
            spec[grid == energy] = spec[grid == energy].mean()
    return grid[~select_duplicate], spec[~select_duplicate]



def get_mse_n2s(specs, window=5, method='simple'):
    '''
    Return the simple moving average of spectra with a rolling window.
    
    Parameters
    ----------
    spec: 2D-array
        The input spectra of shape (N,M) where N and M are the length of data and grid.
    window: int
        The rolling window over which the data is averaged.
    method : str
        The method used to smooth the data.
            `simple`: simple moving average

    Returns
    -------
    (mse, n2s) : tuple(float, float)
        `mse` is the mean square error between the original data and the smoothed data
        `n2s` is the noise to signal value defined by `MSE/var(smoothed_data)`.
    '''
    
    assert not np.isnan(specs).any()

    left = -np.ceil(-window/2)
    right = np.ceil(window/2)
    spec_smoothed = specs.copy()
    for i in range(specs.shape[1]):
        if i>=left:
            spec_smoothed[:,i] = specs[:,int(i-left):int(i+right)].mean(axis=1)
        else:
            spec_smoothed[:,i] = specs[:,i]
    mse = []
    for spec, spec_s in zip(specs, spec_smoothed):
        mse.append(mean_squared_error(spec, spec_s))
    
    mse = np.array(mse)
    n2s = mse/spec_smoothed.std(axis=1)

    return (mse, n2s)



def find_edge(spec, grid, s=1, k=5, initial_guess=4980, method='trust-krylov',
              height=0.1, main_peak_range=(4980,5000)):
    
    '''
    Find the edge of an XAS data by finding the peak position of its derivative.
    
    Return:
    -------
    The optimized result that contains the peak position information
    The analytical spec_function and target_function.
    '''
    # Define a named tuple to query position from Result.x, in line with `minimize`` result.
    Result = namedtuple("Result", "x")
    
    # function for smoothed data and its derivative
    spec_func = UnivariateSpline(grid, spec, s=s, k=k)
    target_func = spec_func.derivative()
    # jacobian and hessian function
    jac_func = lambda x: -target_func.derivative(n=1)(x)
    hess_func = target_func.derivative(n=2)
    
    # determine the initial guess value
    if initial_guess == 'max': # position of maximum peak
        initial_guess = grid[target_func(grid).argmax()]
    elif initial_guess == 'first': # position of first peak
        peak, _ = find_peaks(target_func(grid), height=height)
        initial_guess = grid[peak[0]]
    
    # Use main peak range as a subset of energy range for finding the edge position
    select = (grid>main_peak_range[0]) & (grid<main_peak_range[1])
    grid2min = grid[select]
    
    if method == "simple":
        # simply find the maximum value of the derivative
        peak_pos = np.array([grid2min[np.argmax(target_func(grid2min))]])
        optimize_result = Result(peak_pos)
    else:
        # minimization of the negative of the derivative
        optimize_result = minimize(lambda x: -target_func(x),initial_guess,
                               jac=jac_func, hess=hess_func, method=method)
    
    # return the optimize_result, smoothed spec function and derivative function
    return optimize_result, spec_func, target_func