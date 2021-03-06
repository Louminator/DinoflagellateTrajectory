from scipy import optimize # Necessary for "ret = optimize.basinhopping....." function call
from scipy import *
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def read_data(filename):
    ''' This function reads in x, y, z coordinates and ID for individual points along a
        track from a text file.  The text file should should include only the data points
        to be fit to the helix, starting at time 0.  It returns a tuple of arrays containing
        the positional coordinates to be passed to the basin-hopping algorithm, as well as
        the IDs for graphing if necessary.
    '''
            
    file1 = filename
    
    file1 = open(file1, 'r')
    x = []
    y = []
    z = []
    t = []
    ID = []
    
    for line in file1:
        record = line.rstrip()
        record = record.split('\t')
        
        record[0] = float(record[0])
        x.append(record[0])
        
        record[1] = float(record[1])
        y.append(record[1])

        record[2] = float(record[2])
        z.append(record[2])

        record[3] = float(record[3])
        t.append(record[3])

        record[4] = int(record[4])
        ID.append(record[4])

    file1.close()
    
    x = asarray(x)
    y = asarray(y)
    z = asarray(z)
    t = asarray(t)
    
    data = [x, y, z, t]

    return [tuple(data), ID]

def data_generation(r,g,p, alpha, beta, phi, xo, yo, zo, num_pnts, end_angle, noise_sd):
    '''This function generates data by creating x, y, and z coordinates for a helix
       extending along the z-axis, and calls the rot_trans function to rotate and
       translate the data as the user specifies.
       Parameters:
           r - radius
           g - gamma (angular frequency)
           p - pitch
           phi - phase shift (equivalent to rotation about the z-axis)
           unrot - the unrotated coordinates of a helix
           alpha - the angle rotated about the x-axis
           beta - the angle rotated about the y-axis
           xo, yo, zo - translations in x, y, and z
           num_points - the number points desired
           end_angle - t will be generated as a list of values from 0 to end_angle
           noise_sd - the standard deviation of the normal distribution for adding
                       noise to the data
    '''

    t = linspace(0,  end_angle, num_pnts)
    x = r * sin(g * t + phi)
    y = r * cos(g * t + phi)
    z = g * p / (2 * pi) * t

    random.seed(seed=10)

    rand_x = x + random.normal(scale = noise_sd, size = x.shape)
    rand_y = y + random.normal(scale = noise_sd, size = y.shape)
    rand_z = z + random.normal(scale = noise_sd, size = z.shape)
    
    unrot_data = [rand_x, rand_y, rand_z]

    return rot_trans(unrot_data, alpha, beta, xo, yo, zo) + [t]

def plot_data(data,ID,ax): 
    ''' This function graphs the data, and annotates each point so that the user may
        determine how to split up the helix for analysis.
    '''

    ax.plot(data[0], data[1], data[2])

    # Annotating each point
    for i in arange(0, len(data[0])):
        ax.text(data[0][i], data[1][i], data[2][i],  "%s" % (ID[i]), size=7, zorder=100)

    # Making axes equal range (from http://stackoverflow.com/questions/13685386/
    # matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to)
    
    ax.set_aspect('equal')

    X = data[0]
    Y = data[1]
    Z = data[2]

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax.plot([xb], [yb], [zb], 'w')

    plt.grid()

    plt.xlabel('x ($\mu$m)')
    plt.ylabel('y ($\mu$m)')
    ax.set_zlabel('z ($\mu$m)')

def rot_trans(unrot, alpha, beta, xo, yo, zo):
    '''Rotates the input data about the x and then y axes, then translates.  This
        is primarily used for graphing the solution and generating data.
        Parameters
            unrot - the unrotated coordinates of a helix
            alpha - the angle rotated about the x-axis
            beta - the angle rotated about the y-axis
            xo, yo, zo - translations in x, y, and z
    '''

    Matrix = [unrot[0], unrot[1], unrot[2], ones(size(unrot[2]))]  # ones(size(unrot[2])): creates
                                                                   # an array of ones for translation                                                            

    Ralpha = [[1, 0, 0, 0], [0, cos(alpha), -sin(alpha), 0],           # x rotation
              [0, sin(alpha), cos(alpha), 0], [0, 0, 0, 1]]

    Rbeta = [[cos(beta), 0, sin(beta), 0], [0, 1, 0, 0],               # y rotation
         [-sin(beta), 0, cos(beta), 0], [0, 0, 0, 1]]                  

    T = [[1, 0, 0, xo], [0, 1, 0, yo], [0, 0, 1, zo], [0, 0, 0, 1]]    # Translation

    xR = dot(Ralpha, Matrix)
    yxR = dot(Rbeta, xR)
    TyxR = dot(T, yxR)

    return [TyxR[0], TyxR[1], TyxR[2]]

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

def plot_prelim_angles(z, trans_data,ax):
    '''This function plots the translated data with the optimal vector z, the projected points,
        and the center of mass, allowing the user to visually confirm that z does indeed point
        in the direction of the helix.
        Parameters
            z - the normal vector found by basinhopping
            trans_data - the initially translated data
    '''
    n = size(trans_data[0])  # Number of data points

    pXi = (dot(z, trans_data) / dot(z,z)) 
    
    zs = array(tile(z,(n, 1)))  # Replicate z n times
    
    v = array(map(f,  pXi, zs))  # Multiplies each z by the corresponding scalar from
                                  # the array pXi
                                  
    vt = transpose(v)  # Transpose to fit His ([xxx],[yyy],[zzz])

    Xi = trans_data - vt  # Projected points

    ## Calculate X bar, Y bar, Z bar for the center of mass of the projected points

    Xbar = [sum(Xi[0])/n, sum(Xi[1])/n, sum(Xi[2])/n]

    helix_length = sqrt((trans_data[0][0]-trans_data[0][-1])**2 +
                        (trans_data[1][0]-trans_data[1][-1])**2 +
                        (trans_data[2][0]-trans_data[2][-1])**2)
    ax.plot(trans_data[0], trans_data[1], trans_data[2], 'o', label ='Data')
    ax.plot(Xi[0], Xi[1], Xi[2], 'o', label = 'Projected data')
    ax.plot([Xbar[0],Xbar[0]+.2*helix_length*z[0]], [Xbar[1],Xbar[1]+.2*helix_length*z[1]],
            [Xbar[2],Xbar[2]+.2*helix_length*z[2]], 'r', label = 'Normal vector')
    ax.plot([Xbar[0]], [Xbar[1]], [Xbar[2]], 'o', label = 'Center of mass')

    # Making axes equal range (from http://stackoverflow.com/questions/13685386/
    # matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to)

    ax.set_aspect('equal')

    X = trans_data[0]
    Y = trans_data[1]
    Z = trans_data[2]

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax.plot([xb], [yb], [zb], 'w')

    plt.grid()

    plt.xlabel('x ($\mu$m)')
    plt.ylabel('y ($\mu$m)')
    ax.set_zlabel('z ($\mu$m)')
    ax.legend(numpoints=1,fontsize=11)
    
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

def plot_solution(r, g, p, alpha, beta, phi, xo, yo, zo, data,ax):
    '''This function provides a way to visually confirm the fit.  It graphs the
        data points along with a helix using the parameters found to best fit the data.
        Parameters:
            r, g, p, alpha, beta, phi, xo, yo, zo - helix parameters found with basinhopping
            data - the data 
    '''
    
    t = linspace(0, data[3][-1], 100)
    xs = r * sin(g * t + phi)
    ys = r * cos(g * t + phi)
    zs = g * p / (2 * pi) * t

    solution = [xs, ys, zs]

    solution = rot_trans(solution, alpha, beta, xo, yo, zo) + [t]

    ax.plot(solution[0], solution[1], solution[2], label = 'Solution')

    ax.plot(data[0], data[1], data[2], 'o', label = 'Data')

    # Making axes equal range (from http://stackoverflow.com/questions/13685386/
    # matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to)
    
    ax.set_aspect('equal')

    X = data[0]
    Y = data[1]
    Z = data[2]

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax.plot([xb], [yb], [zb], 'w')

    plt.grid()
    
    plt.xlabel('x ($\mu$m)')
    plt.ylabel('y ($\mu$m)')
    ax.set_zlabel('z ($\mu$m)')
    ax.legend(fontsize=11)

###############################################################################################
# Function calls
#
# Note: See function descriptions for more complete explanations
# 
##################
# Data source
#
# For simulated data, uncomment this line:
#
#data = data_generation(4,1,10,.5,1,0,20,30,40,10,4*pi,.01)
#
# (r, g, p, alpha, beta, phi, xo, yo, zo, num_pnts, end_angle, noise_sd)
#
# The data_generation function simulates data using the parameters listed above.
# Parameters should be changed as the user sees fit.
#
#
# For real data, uncomment the following lines:
#
full_data = read_data('KV_7.22_120fps_2_Track1519_full.txt')
origdata = array(full_data[0])
origID = full_data[1]

#Trim the data to a smaller size

#data = origdata[:,:12]
#ID = origID[:12]
#
###################

for start in range(0,17):

    data = origdata[:,start:start+8]
    
    # Shift time to zero for the first data point.
    data[3,:] = data[3,:]-data[3,0]
    ID = origID[start:start+8]
    
    fig = plt.figure()
    
    # To plot the data with point annotation
    ax = fig.add_subplot(221,projection='3d')
    plot_data(data,ID,ax)
    
    # This function performs a rough translation to the origin
    trans_data = prelim_params_trans(data)
    
    # This function calls the basinhopping algorithm to find preliminary parameter guesses
    [r_guess, beta_guess, alpha_guess, z] = call_bh_prelim_params(trans_data)
    
    # This is probably a good idea, but we do not konw for sure if it is necessary.
    if (alpha_guess<0):
        alpha_guess = alpha_guess+2*pi
    if (beta_guess<0):
        beta_guess = beta_guess+2*pi
    
    # This plots the translated data and the projected data to ensure that the normal to
    # the plane of projection (z) found by basinhopping does indeed point in the direction
    # of the helix.
    # NOTE: Be sure to close the plot, otherwise the script will not continue to evaluate
    ax = fig.add_subplot(222,projection='3d')
    plot_prelim_angles(z, trans_data,ax)
    
    # This calls the main basinhopping algorithm, and returns the helical parameters best fit to the data
    [ r, g, p, alpha, beta, phi, xo, yo, zo, main_epsilon ] = call_bh_main(r_guess, alpha_guess, beta_guess, data)
    
    # This provides a visual check for by plotting the solution helix with the data.
    ax = fig.add_subplot(223,projection='3d')
    plot_solution(r, g, p, alpha, beta, phi, xo, yo, zo, data,ax)
    
    # NOTE: If the main optimization is failing to find the correct helix, try switching sin
    # and cos in main_helix_opt and plot_solution

plt.show()