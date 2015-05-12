from scipy import optimize # Necessary for "ret = optimize.basinhopping....." function call
from scipy import *


def prelim_params_trans(data):
    '''This function makes a rough translation of the data points back to the origin,
       and puts the data in a form that can be passed to the basinhopping algorithm.
       Parameters
           data - data to be optimized
    '''
    x = data[0]                         
    trans_x = x - x[0] # translates points to the origin by subtracting the first x coordinate
                        # from the vector of all x coordinates
    y = data[1]
    trans_y = y - y[0]

    z = data[2]   
    trans_z = z - z[0]

    trans_data = [asarray(trans_x), asarray(trans_y), asarray(trans_z)]
    trans_data = tuple(trans_data)    # data to send to basinhopping must be a tuple

    return trans_data

def f(pXi, zs):
    '''This function is used by the prelim_params_test function to multiply vectors within
       a vector (zs) by scalars within another vector (pXi)
    '''
    return(pXi*zs)

def prelim_params_test(x,*args):
    '''This is the function passed to basinhopping which returns the sum of the distances from
       the center of mass of points projected onto a plane.  The normal vector to the plane is
       dependent on two angles, which are the parameters to be optimized.
       Parameters
           x - array of parameters to be optimized
           *args - contains the data coordinates.  It is passed to this function through
                minimizer_kwargs when calling the basin-hopping function.
                The data is called trans_data and is in the form [x array, y array, z array]
    '''

    # x[0] = theta
    # x[1] = psi
    # z is the normal vector to the plane, and is defined by spherical coordinates with a
    # radius of 1 and angles theta and psi.
    
    z = [cos(x[0])*sin(x[1]), sin(x[0])*sin(x[1]), cos(x[1])]

    n = size(trans_data[0])  # Number of data points

    pXi = (dot(z, trans_data) / dot(z,z)) 
    
    zs = array(tile(z,(n, 1)))  # Replicate z n times
    
    v = array(map(f,  pXi, zs))  # Multiplies each z by the corresponding scalar from
                                  # the array pXi
                                  
    vt = transpose(v)  # Transpose to fit His ([xxx],[yyy],[zzz])

    Xi = trans_data - vt  # Projected points

    ## Calculate X bar, Y bar, Z bar for the center of mass of the projected points

    Xbar = [sum(Xi[0])/n, sum(Xi[1])/n, sum(Xi[2])/n]

    epsilon = sum((Xi[0] - Xbar[0]) ** 2 +    # The sum of the distances from the COM
                   (Xi[1] - Xbar[1]) ** 2 + 
                   (Xi[2] - Xbar[2]) ** 2)
    return epsilon

    
def call_bh_prelim_params(trans_data):
    '''This function calls the basinhopping algorithm to minimize the sum of the distances
        between the data points projected onto an arbitrary plane, and the center of mass
        of the projected points
        Parameters
            trans_data - data roughly translated to the origin
    '''
    
    minimizer_kwargs = {"method": "L-BFGS-B", "args": trans_data, "bounds": ((0,2*pi),(0,2*pi))}
    
    x0 = [pi, pi]
    
    ret = optimize.basinhopping(prelim_params_test, x0, minimizer_kwargs=minimizer_kwargs, niter = 200)
    print("Preliminary parameters minimization: x = [%.4f, %.4f], epsilon = %.4f" %\
          (ret.x[0], ret.x[1], ret.fun))

    z = array([cos(ret.x[0])*sin(ret.x[1]), sin(ret.x[0])*sin(ret.x[1]), cos(ret.x[1])])

    epsilon = ret.fun

    n = size(trans_data[0])

    r_guess = sqrt(epsilon / n )         # average distance from COM
    beta_guess = pi - arctan2(-z[0],z[2])
    alpha_guess = arctan2(z[1], sqrt((z[0])**2 + (z[2])**2))
    print('Initial guess for alpha, beta and r from preliminary parameter test:')
    print('alpha = %.4f' %alpha_guess)
    print('beta = %.4f' %beta_guess)
    print('r = %.4f' %r_guess)

    return r_guess, beta_guess, alpha_guess, z

def main_helix_opt(x,*args):
    '''This function primarily acts as the function to be passed to the basin hopping
    optimization.  It calculates and returns the error between the data coordinates
    and the calculated coordinates (including rotation and translation).
    Parameters
        x - array of parameters to be optimized
        *args - contains the data coordinates.  It is passed to this function through
                minimizer_kwargs when calling the basin-hopping function.
    '''
    
    # x[0] = r
    # x[1] = g
    # x[2] = p
    # x[3] = alpha
    # x[4] = beta
    # x[5] = phi
    # x[6] = xo
    # x[7] = yo
    # x[8] = zo
    # data[3] = t

    X = x[0] * sin(x[1] * data[3] + x[5])   # X = rsin(gt + phi)
    Y = x[0] * cos(x[1] * data[3] + x[5])   # Y = rcos(gt + phi)
    Z = x[1] * x[2] / (2 * pi) * data[3]    # Z = gp/(2pi)*t
    A = x[3]                                # A = alpha
    B = x[4]                                # B = beta
    
    epsilon = sum((data[0] - (cos(B) * X + sin(B) * (sin(A)*Y + cos(A)*Z) + x[6])) ** 2 +
               (data[1] - (cos(A)*Y - sin(A)*Z + x[7])) ** 2 + 
               (data[2] - (-sin(B)*X + cos(B)*(sin(A)*Y + cos(A)*Z) + x[8])) ** 2)
    
    return epsilon
    
def call_bh_main( r_guess, alpha_guess, beta_guess, data):
    '''This function calls the basinhopping algorithm to minimize the main function to
        be optimized, that is the error between the data points and the equation for a
        helix.
        Parameters
            r_guess - the guess for r found from the preliminary parameters test
            alpha_guess
            beta_guess
            data - full original data
    '''
    
    # Initial guesses.
    # data[0][0] uses the first x coordinate as the initial guess for the x translation.
    # x0 = [r, g, p, alpha, beta, phi, xo, yo, zo]
    x0 = [r_guess, 4, 10, alpha_guess, beta_guess, 0, data[0][0], data[1][0], data[2][0]] 


    # Timing the algorithm
    import time
    start_time = time.time()

    # Additional arguments passed to the basin hopping algorithm.  The method chosen allows
    # bounds to be set.  The first bounds element is the bounds for the radius, and should be
    # estimated from the initial graphing of the data.  The third element is the pitch, and
    # should also be estimated.  The last three elements are the bounds for the translations,
    # and are estimated automatically from the first x, y, and z coordinates of the data set.

    minimizer_kwargs = {"method": "L-BFGS-B", "args": data, "bounds": ((r_guess*0.5,r_guess*1.5),(0,2*pi),(0, None),
                                                     (0,2*pi),(0,2*pi),(0,2*pi),
                                                                       (data[0][0] - 20, data[0][0] + 20),
                                                                       (data[1][0] - 20, data[1][0] + 20),
                                                                       (data[2][0] - 20, data[2][0] + 20))}

    ret = optimize.basinhopping(main_helix_opt, x0, minimizer_kwargs=minimizer_kwargs, niter = 200)
    print('')
    print("Main solution parameters: r = %.4f, g = %.4f, p = %.4f, alpha = %.4f, beta = %.4f, phi = %.4f,\
 xo = %.4f, yo = %.4f, zo = %.4f], epsilon = %.4f" %\
          (ret.x[0], ret.x[1], ret.x[2], ret.x[3], ret.x[4], ret.x[5], ret.x[6], ret.x[7], ret.x[8], ret.fun))

    seconds = time.time() - start_time
    minute = seconds // 60
    sec = seconds % 60
    print('Time taken to find global minimum:')
    print("--- %.2f minutes, %.2f seconds ---" % (minute, sec))
    print("--- %.2f seconds ---" % seconds)

    return ret.x[0], ret.x[1], ret.x[2], ret.x[3], ret.x[4], ret.x[5], ret.x[6], ret.x[7], ret.x[8], ret.fun
