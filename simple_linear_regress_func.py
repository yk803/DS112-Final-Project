# This function computes the SLR for input 'data'
# It assumes that 'data' has one column for regressor and one for predictor
# It returns a numpy array 'output' containing: m, b and r_sqr

def simple_linear_regress_func(data):
    # Load numpy:
    import numpy as np
    # Initialize container:
    temp_cont = np.empty([len(data),5]) # initialize empty container, n x 5
    temp_cont[:] = np.NaN # convert to NaN
    # Store data in container:
    for i in range(len(data)):
        temp_cont[i,0] = data[i,0] # x 
        temp_cont[i,1] = data[i,1] # y 
        temp_cont[i,2] = data[i,0]*data[i,1] # x*y
        temp_cont[i,3] = data[i,0]**2 #x^2
        temp_cont[i,4] = data[i,1]**2 # y^2
    # Compute numerator and denominator for m:
    m_numer = len(data)*sum(temp_cont[:,2]) - sum(temp_cont[:,0])*sum(temp_cont[:,1])
    m_denom = len(data)*sum(temp_cont[:,3]) - (sum(temp_cont[:,0]))**2
    m = m_numer/m_denom
    # Compute numerator and denominator for b:
    b_numer = sum(temp_cont[:,1]) - m * sum(temp_cont[:,0])
    b_denom = len(data)
    b = b_numer/b_denom
    # Compute r^2:
    temp = np.corrcoef(data[:,0],data[:,1]) # pearson r
    r_sqr = temp[0,1]**2 # L-7 weenie it (square)
    # Output m, b & r^2:
    output = np.array([m, b, r_sqr])
    return output 