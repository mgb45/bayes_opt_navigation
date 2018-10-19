from __future__ import print_function
from numpy import array,matrix,eye,ones,isnan,isinf,average,sqrt,matmul


__author__ = "Jaco Jansen van Rensburg"
__copyright__ = "Copyright 2018, Jaco Jansen van Rensburg"
__license__ = "GPL"
__version__ = "1.0.2"
__email__ = "jjvrensburg@csir.co.za, jacojvrensburg@gmail.com"


class rbf:
    '''

*******************************************************************************

    A SIMPLE RADIAL BASIS FUNCTION INTERPOLATOR / REGRESSOR:
    
*******************************************************************************
    
    This object takes a multi dimensional input [ x_i ]
                      a one dimensional outpt   [ f_i ] = f(x_i)
                      
    to approximate the function  
                         
         f(x) ~  S(x) =  SUM   [  beta  x phi (x)  + mu  ]
                            k         k      k
                            
    with [ beta ] the kernel weights and [ phi ] the kernel functions.
                            
    The object will regress if fewer RBF kernels [ x_k ] than samples [ x_i ]
    are specified.
    
    The RBF approximated gradient :  d        
                                    --- S(x)
                                     dx
    and variance of the fit / statistical error : ERROR ( S(x) ) are also avalable.
                           
                           
    
*******************************************************************************

    '''
    def __init__(self,x_i,f_i,**kwargs):
        '''
            
    >>  Initialise an object to create an RBF surrogate model relating 
        
        x_i [inputs] --> f_i [outputs]
        
        where f_i = f(x_i) of some unknown continuous function f(x)
    
    
    Parameters:
    
        x_i : array_like
            Input value array of shape [M]x[N]
            i.e. [M] observations in [N] dimensional space
            
        f_i : array_like
            Output value array of shape [M]x[1]
            i.e. [M] observations of a single value output function f(x_i)
    
    
    
    >>  Possible keyword arguments and default values:
    
        kernels  : array_like
            Kernel value array of shape [K]x[N] that define the RBF kernel
            locations. If None, the input values [x_i] are used as
            kernel locations.
            
            [DEFAULT] = None
        
        
        function : string
            Use a set of pre-defined kernel functions [ phi(r,d) ]
            
            'gaussian' :
                phi(r,d)  =  exp(-(r/d)**2)
               
            'thin_plate':
                phi(r,d)  =  0  if r==0  else (r/d)**2*log(r/d)
                
            'cubic':
                phi(r,d)  =  (r/d)**3
                
            'multiquadric':
                phi(r,d)  =  sqrt(1+(r/d)**2)
                
            'inverse_quadric':
                phi(r,d)  =   1/(1+(r/d)**2)
                
            'inverse_multiquadric':
                phi(r,d)  =  1/sqrt(1+(r/d)**2)
                
            'spherical_harmonic':
                phi(r,d)  =  2*exp(-r/d)/(1+exp(-2*r/d))
                
            'compact_support':
                phi(r,d)  =  0 if r>d  else (1-r/d)**4*(4*r/d+1)
                
                
            [DEFAULT] = 'gaussian'
                
        f : callable
            A user defined kernel function
            *** only needed if not using one of the built-in functions
            Example, for 'gaussian':
            
            f = lambda r,d=1 : exp(-(r/d)**2)
            
            [DEFAULT] = None
           
           
        df : callable
            A user defined kernel function derivative
            *** only needed if not using one of the built-in functions
            Example, for 'gaussian':
            
            df = lambda r,d=1 : -2*r*exp(-(r/d)**2)/d**2
            
            *** if [ f ] is a user-defined function but no [ df ] provided,
            it is approximated using finite differences
            
            [DEFAULT] = None
            
            
        epsilon : float
            The kernel length scale / support radius
            
            if epsilon is None, it defaults to the average distance between
            samples [x_i] which is usually a reasonable initial estimate.
            
            [DEFAULT] = None
            
        scalef : float OR array_like
            A user defined input variable scale function
            
            x_k = x_k*scalef
            
            [DEFAULT] = 1
        
        average : boolean
            Work around average sample input
            
            x_k = x_k - average(x_k)
            
            [DEFAULT] = False
            
        standardise : boolean
            Use the input variable standard deviation for dimensional scale
            
            x_k = x_k/std(x_k)
            
            [DEFAULT] = False
            
        radial_tolerance : float
            A tolerance value between the radial distances of RBF kernels.
            If a new / candidate kernel is investigated, and falls within the 
            radial tolerance of existing kernels, it is ignored.
            
            Done to prevent linear dependence in the kernel / covariance
            matrix.
            
            [DEFAULT] = 1e-8
            
        smooth : float
            Tikhonov regularisation parameter, used to smooth the RBF
            response or to lesten matrix ill-conditioning.
            If an ill-conditioned matrix is detected, the parameter is 
            automatically increased:
            smooth = 1e-10 or smooth = smooth*10
            untill maximum allowable matrix condition number is no longer
            violated
            
            [DEFAULT] = 0.
            
        max_condition : float
            Maximum allowable matrix condition number.
            Checked for possible ill-conditioning
            
            [DEFAULT] = 1e16
            
        regress_solution : string
            Solution to the regression preformed by either 'least squares'
                                                        or 'pseudo-inverse'
            
            [DEFAULT] = 'pseudo-inverse'
        
        '''
        
        self.kwdict = {
                  # kernel locations (if "None" then the X values will be kernel locations)
                  'kernels':None,
                  # Choose from a predefined function 
                  'function':None,
                  # set a callable basis function f(r,epsilon)
                  'f':None,
                  # set the callable derivative of the basis function d(r,epsilon)/dr
                  'df':None,
                  # set the radius of influence value in f(r,epsilon)
                  'epsilon':None,
                  # set the input dimension scale factor (or scale factor per dimension)
                  'scalef':1.,
                  # construct the RBF approximation in averaged input dimension space
                  'average':False,
                  # construct the RBF approximation in standardised input dimension space (nullidies the scalef value)
                  'standardise':False,
                  # tolerance in the calculated radius 
                  # > i.e. if a point is within this distance from a previous point, it is used as an observation only (not a kernel)
                  'radial_tolerance':1e-8,
                  # polynomial order
                  # NOTE: Arbitrary dimensional polynomial currently ignored in function and gradient approximation 
                  'polynomial':-1, 
                  # smoothness parameter (Tikhonov regularisation)
                  'smooth':0.,
                  # the maximum allowable condition number (if violated, a smoothness parameter is applied / increased)
                  'max_condition':1e16,
                  # option taken for rectangular matrices (either "pseudo-inverse" or "least-squares"
                  'regress_solution':'pseudo-inverse'
                  }
        
        # transfer information from **kwargs to kwdict:
        unknown_keyword=False
        for kw in kwargs.keys():
            if kw in self.kwdict.keys():
                self.kwdict[kw] = kwargs[kw]
            else:
                print('Could not recognise keyword << %s >>'%kw)
                unknown_keyword = True
                
        if unknown_keyword:
            prnt_help = input('\t\t>> print help? [N]/y ').lower() or 'n'
            if prnt_help[0] == 'y':
                print(self.__init__.__doc__)
        
        #
        #keywords = ['kernels',
                    #'function',
                    #'f',
                    #'df'
                    #'epsilon',
                    #'scalef',
                    #'average',
                    #'standardise',
                    #'radial_tolerance',
                    #'smooth',
                    #'max_condition']
        #
        # set / scale the sample observations
        self._set_samples(x_i,f_i)
        #
        # set the RBF functions
        self._set_funct()
        #
        # calculate the weights
        self._calc_weights()
        #
        
        
        
        
        
        
        
        
        
        
    def _set_samples(self,x_i,f_i,**kwargs):
        '''
        
    Take the observations 
        
        f_i = f(x_i) 
        
    and construct distance matrix using specific kernel locations 
        
        x_k = kernels   ... if 'kernels' are defined
        x_k = x_i       ... otherwise
        
        
    Parameters:
    
        x_i : array_like
        
            Input value array of shape [M]x[N]
            i.e. [M] observations in [N] dimensional space
            
        f_i : array_like
        
            Output value array of shape [M]x[1]
            i.e. [M] observations of a single value output function f(x_i)
    
    
    
        NOTE: The input dimension is modified according to either:
        
        scalef  :
            [1] OR [N] dimensional scale factor
            i.e. x_k /= scalef
            
        average :
            x_avg = average(x_k) along dimension [M]
            x_k -= x_avg
            
        standardise :
            x_std = standard deviation(x_k) along dimension [M]
            x_k /= x_std
        
        '''
        
        #
        for kw in kwargs.keys():
            if kw in self.kwdict.keys():
                self.kwdict[kw] = kwargs[kw]
        #
        # some useful functions:
        from scipy.spatial.distance import cdist
        from numpy import array, average, std, isfinite
        
        #
        # make sure the input array is of the correct type and shape:
        x_i = array(x_i)
        if x_i.shape.__len__() == 1: x_i = x_i.reshape(-1,1)
        self._m,self._n = x_i.shape
        
        # make sure the output array is of the correct type and shape:
        self.f_i = array(f_i).reshape(-1,1)
        
        # keyword dictionary values:
        kernels           = self.kwdict['kernels']
        scalef            = self.kwdict['scalef']
        average           = self.kwdict['average']
        standardise       = self.kwdict['standardise']
        radial_tolerance  = self.kwdict['radial_tolerance']
        
        #
        # check if input dimensions should be averaged
        if average:
            self.x_avg = average(x_i,0)
        else:
            self.x_avg = 0.
        #
        # check if input dimensions should be scaled / standardised:
        x_sf = array([[1]])
        if not scalef is None:
            x_sf = array(scalef).reshape(-1,1)
            x_sf[x_sf==0] = 1
        if standardise:
            x_sf = std(x_i,0)
            x_sf[x_sf==0] = 1
            x_sf = 1./x_sf
        #
        x_sf[isfinite(x_sf)==False] = 1
        self.x_sf = x_sf
        #
        # modify the input array:
        self.x_i = self.x_sf*(x_i-self.x_avg)
        #
        # do kernels / initial kernel approximates
        use_x_i = True
        if not kernels is None:
            x_k = array(kernels)
            if x_k.shape.__len__() == 1: x_k = x_k.reshape(-1,1)
            _k, _n2 = x_k.shape
            if _n2 == self._n:
                x_k = self.x_sf*(x_k-self.x_avg)
                use_x_i = False
        if use_x_i:
            x_k = self.x_i
            _k = self._m  # initial estimate on the number of kernels (subject to radial_tolerance)
        #
        # make sure the kernels are all outside allowable inter-kernel radius to prevent linear dependece:
        self.x_k = [x_k[0]]
        for i in range(1,_k): 
            if all(cdist(self.x_k,x_k[i].reshape(1,-1))>radial_tolerance):
                self.x_k += [x_k[i]]
        self.x_k = array(self.x_k)
        self._k = self.x_k.shape[0]
        
        #
        # calculate pairwise distances / radii between x_i and x_k
        self.radii = cdist(self.x_i,self.x_k,'euclidean')
        
        
        
        
        
        
        
        

       
    def _set_funct(self,**kwargs):
        '''
        
        Set the kernel function and associated properties
        
        '''
        
        for kw in kwargs.keys():
            if kw in self.kwdict.keys():
                self.kwdict[kw] = kwargs[kw]
                
        # useful functions
        from numpy import average, log, exp, sqrt
        
        
        # specific kewords of interest:
        self.function = self.kwdict['function']
        f             = self.kwdict['f']
        df            = self.kwdict['df']
        epsilon       = self.kwdict['epsilon']
        
        # if epsilon is undefined, use the average radial distance:
        if epsilon is None:
            try:
                # do we already have an epsilon defined?
                epsilon = self.epsilon
            except:
                # use the average radial distance (usually a good first extimate)
                epsilon = average(self.radii)
            
        
        d = self.epsilon = epsilon
            
        # see if own function defined
        if callable(f):
            
            self.f = lambda r,d=d  : f(r,d)
            
            if self.function is None:
                self.function = 'user_defined'
                
            if callable(df) : 
                self.df = lambda r,d=d  : df(r,d)
            else:
                # use a central finite difference approximate
                self.df = lambda r,d=d : (f(r+1e-6,d)-f(r-1e-6,d))/2e-6
            
        
        # possible pre-defined functions:
        if not callable(f):
            
            
            if self.function == 'thin_plate':
                self.f   =  lambda r,d=d  :  0  if r==0  else (r/d)**2*log(r/d)
                self.df  =  lambda r,d=d  :  0  if r==0  else (2*log(r/d) + 1)*r/d**2
                
            elif self.function == 'cubic':
                self.f   =  lambda r,d=d  :  (r/d)**3
                self.df  =  lambda r,d=d  :  3*(r/d)**2
                
            elif self.function == 'multiquadric':
                self.f   =  lambda r,d=d  :  sqrt(1+(r/d)**2)
                self.df  =  lambda r,d=d  :  r/(d**2*sqrt(1+(r/d)**2))
                
            elif self.function == 'inverse_quadric':
                self.f   =  lambda r,d=d  :  1/(1+(r/d)**2)
                self.df  =  lambda r,d=d  :  -2*r/(d*(1+(r/d)**2))**2
                
            elif self.function == 'inverse_multiquadric':
                self.f   =  lambda r,d=d  :  1/sqrt(1+(r/d)**2)
                self.df  =  lambda r,d=d  :  -r/(d**2*(1+(r/d)**2)**1.5)
                
            elif self.function == 'spherical_harmonic':
                self.f   =  lambda r,d=d  :  2*exp(-r/d)/(1+exp(-2*r/d))
                self.df  =  lambda r,d=d  :  4*exp(-3*r/d)/(d*(1+exp(-2*r/d))**2) - 2*exp(-r/d)/(d*(1+exp(-2*r/d)))
                
            elif self.function == 'compact_support':
                self.f   =  lambda r,d=d  :  0 if r>d  else (1-r/d)**4*(4*r/d+1)
                self.df  =  lambda r,d=d  :  0 if r>d  else 4*((1-r/d)**4-(1-r/d)**3*(4*r/d+1))/d 
                
            else:   
                self.function = "gaussian"
                self.f   =  lambda r,d=d  :  exp(-(r/d)**2)
                self.df  =  lambda r,d=d  :  -2*r*exp(-(r/d)**2)/d**2
            
            
            
            
            
            
            
    def _set_smoothing(self,PHI,iprint=0):     
        
        from numpy import eye
        from numpy.linalg import cond     
            
        smooth = self.kwdict['smooth']
        if smooth is None: smooth = 0.
        
        Keye = eye(PHI.shape[0])
        
        PHI += smooth*Keye
        condnr = cond(PHI)
        
        smooth_prev = smooth
        altersmooth = False
        while condnr > self.kwdict['max_condition']:
            if smooth<1e-10 : smooth = 1e-11
            smooth *= 10
            dsmooth = smooth-smooth_prev
            smooth_prev = smooth
            PHI += dsmooth*Keye
            condnr = cond(PHI)
            altersmooth = True
            
        if altersmooth:
            if iprint:
                print('<<  WARNING  >>\n\t\t Altered Tikhonov smoothing to %e \n\t\t Matrix condition number = %e < %e maximum allowed'%(smooth,condnr,self.kwdict['max_condition']))
            #self.kwdict['smooth'] = smooth
            
        return PHI
            
            
            
            
            
            
            
            
    def _calc_weights(self,iprint=1,**kwargs):
        '''
        
        calculate the kernel / polynomial weights
        
        '''
        
        #
        
        for kw in kwargs.keys():
            if kw in self.kwdict.keys():
                self.kwdict[kw] = kwargs[kw]
                
                
        from numpy import array,matrix,eye,ones,isnan,isinf,average,sqrt,matmul
        from numpy.linalg import cond
        from scipy import linalg
        
        
        # covariance matrix:
        Cov = array([self.f(r) for r in self.radii.flatten()]).reshape(-1,self._k)
        
        
        if cond(Cov) > self.kwdict['max_condition']:
            print('<<  WARNING  >>\n\t\t Possible ill conditioning in covariance matrix while calculating kernel weights')
    
            
            
        #
        #
        # CALCULATION
        # Is the error / uncertainty approximation available?
        self.err_available = True
        self.ones = matrix(ones(self._k)).T
        #
        if not self._k ==self._m :
            
            PHI = Cov.T*matrix(Cov)
            f_k = Cov.T*matrix(self.f_i)
            
            PHI = self._set_smoothing(PHI,iprint=iprint)
            
            if 'least' in self.kwdict['regress_solution'].lower():
                # set up [PHI]{w_k} = {f_k}  to reprensent least squares regression
                               
                #
                self.PHIinv = linalg.inv(PHI)
                self.f_k = f_k
                
                #
                self.P_f_k  = self.PHIinv*self.f_k
                self.oPo = self.ones.T*self.PHIinv*self.ones
                
                # f_mean
                self.f_mu = 0
                # modified 
                self.f_mod = self.f_k - self.f_mu
                
                # weights
                w_k = self.P_f_k
                # check the weighting for NaNs and INFs
                w_k[isnan(w_k)]=0.
                w_k[isinf(w_k)]=0.
                self.w_k = array(w_k).flatten()
                # f_s2
                self.f_s2 = (self.f_mod.T*w_k)[0,0]/self._m
            
                
            else:
                #
                # print(" DO inv[sqrt(A.t x A)]")
                # do inv(sqrt(A.t x A)) modification
                U,S,V = linalg.svd(PHI, full_matrices=True)
                
                Sinv = eye(S.size)/sqrt(S)
                
                self.PHIinv = matmul(V.T,matmul(Sinv,U.T))
                self.f_k = matmul(self.PHIinv,f_k)
                
                #
                self.P_f_k  = self.PHIinv*self.f_k
                self.oPo = self.ones.T*self.PHIinv*self.ones
                
                # f_mean
                self.f_mu = 0
                # modified 
                self.f_mod = self.f_k - self.f_mu
                # weights
                w_k = self.P_f_k
                # check the weighting for NaNs and INFs
                w_k[isnan(w_k)]=0.
                w_k[isinf(w_k)]=0.
                self.w_k = array(w_k).flatten()
                # f_s2
                self.f_s2 = (self.f_mod.T*w_k)[0,0]/self._m
                
        else:
            # solve directly
            
            PHI = self._set_smoothing(matrix(Cov),iprint=iprint)
            
            f_k = matrix(self.f_i)
            #
            # invert the matrix
            self.PHIinv =  linalg.inv(PHI)
                
            self.f_k = f_k
            
            self.P_f_k  = self.PHIinv*self.f_k
            self.oPo = self.ones.T*self.PHIinv*self.ones
            
            #
            # f_mean
            self.f_mu = (self.ones.T*self.P_f_k/self.oPo)[0,0]
            # modified 
            self.f_mod = self.f_k - self.f_mu
            
            #
            # weights
            w_k = self.PHIinv*self.f_mod
            # check the weighting for NaNs and INFs
            w_k[isnan(w_k)]=0.
            w_k[isinf(w_k)]=0.
            self.w_k = array(w_k).flatten()
            # f_s2
            self.f_s2 = (self.f_mod.T*w_k)[0,0]/self._m
        
        
        
        
        
        
        
        
        
        

    def __call__(self,x,gradient=False,error=False):
        
        '''
            
    >>  Return an approximate to the the function value using the RBF 
        surrogate model
                         
         f(x) ~  S(x) =  SUM   [  beta  x phi (x)  + mu  ]
                            k         k      k
                            
                               
    
    Parameters:
    
        x : array_like
            Input variable prediction array of shape [M]x[N]
            i.e. 
            return [M] predictions of the approximated [N] dimensional space    
    
    
    >>  Possible keyword arguments and default values:
        
        gradient : boolean
            Return the approximated function gradient(s) = grad[ S(x) ]
            
        error : boolean
            Return the estimmated prediction error(s) = RMSE ( S(x) ) 
            
    >>  Example:
        
        
        f = lambda x: sin(8*x) + 0.66*cos(18*x) + 3 - 2*x
        
        x_i = linspace(0,1,5)
        f_i = f(x_i)
        
        
        f_rbf = surrogate.rbf(x_i,f_i)
        
        x = array([0.1,0.2,0.8]).reshape(-1,1)
        
        fx = f_rbf(x)
        fx, dfdx = f_rbf(x, gradient=True)
        fx, err = f_rbf(x, error=True)
        fx, dfdx, err = f_rbf(x, gradient=True, error=True)    
    
    
        
        '''
        from numpy import array,sqrt,sum,where,isnan,isinf,matrix,matmul
        
        # make sure that new x values are in the correct format and correctly modified:
        x = self.x_sf*(array(x).reshape(-1,self._n)-self.x_avg)
        
        f   = []
        df  = []
        err = []
        derr = []
        
        cor0 = self.f(0)
        
        for x_new in x:
            #
            xdist = self.x_k - x_new
            dists = sqrt(sum(xdist*xdist,1))
            rads = self.f(dists)#array([self.f(r) for r in dists])
            f_comps = rads*self.w_k 
            f += [sum(f_comps)+ self.f_mu]
            
            
            if gradient:
                #xdist = self.x_k - x_new
                ddxlst = -xdist.T/dists#array([-xi/dists for xi in xdist.T])
                drads = self.df(dists)#array([self.df(r) for r in dists])
                df_comps = array([array(self.w_k).flatten()*array(drads).flatten()*array(dx).flatten() for dx in ddxlst]).T
                df_comps[isnan(df_comps)]=0.
                df_comps[isinf(df_comps)]=0.
                df += [sum(df_comps,0)*self.x_sf]
                
                
            if error:
                # from DR Jones Schonlau and Welch
                ri = matrix(rads).T
                PHIvarphi = self.PHIinv*ri
                s2 = self.f_s2*(cor0-ri.T*PHIvarphi+(cor0-self.ones.T*PHIvarphi)**2/self.oPo)[0,0]
                #
                if isnan(s2): s2=0
                err += [max([s2,0])]
                
                
                
                if gradient:
                    
                    # also return the gradient of the variance estimate
                    
                    d_ri = matrix([array(drads).flatten()*array(dx).flatten() for dx in ddxlst]).T
                    d_ri[isnan(d_ri)]=0.
                    d_ri[isinf(d_ri)]=0.
                    
                    
                    PHI_dvarphi = self.PHIinv*d_ri
                    
                    d_s2 = self.f_s2*(-array(PHIvarphi.T*d_ri).flatten()-
                                     array(ri.T*PHI_dvarphi).flatten()-
                                     2*(cor0-self.ones.T*PHIvarphi)[0,0]*
                                     array(self.ones.T*PHI_dvarphi).flatten()/self.oPo[0,0])
                
                    d_s2[isnan(d_s2)] = 0
                    d_s2[isinf(d_s2)] = 0
                    
                    # added the input dimension scalar dependence to the gradient  
                    
                    derr += [d_s2*self.x_sf]
                    
                
        if gradient&error:
            return array(f), array(df), array(err), array(derr)   
        
        if gradient:
            return array(f), array(df)
        
        if error:
            return array(f), array(err)
        
        return array(f)
    
        
     
     
     
     
          
        
        

    def cross_validate(self,epsilon=None,bins=None,randomise=False,iprint=0):
        '''
        
    >>  Calculate the cross-validation error for a specific value of
        [ epsilon ] 
       
    >>  Possible keyword arguments and default values:
        
        epsilon : float
            The kernel function length scale
            If None, the current value is evaluated
            
        bins : integer
            The number of bins / surrogate models constructed to do 
            cross-validation.
            If none, a full "leave-one-out" cross-validation is done
            
        randomise : boolean
            Choose from the sample points at random.
            If [ True  ] a list for radom selection is generated.
            If [ False ] the existing random list is re-used
                         and generated once off if not available
            
        return_errors : boolean
            return the cross-validation errors per sample point
            
        
        '''
        
        #
        # get the list of randomised indices:      
        if randomise:
            from numpy import random
            randlst = random.random(self._k).argsort()
            self.randlst = randlst
        else:
            # use the already defined random list
            try:
                randlst = self.randlst
            except:
                from numpy import random
                randlst = random.random(self._k).argsort()
                self.randlst = randlst
        
        self.f_xve = 0*self.f_i.copy()
                
        if epsilon is None:
            epsilon = self.epsilon
            
        if bins is None:
            bins = self._m # do full leave one out cross-validation
        if bins > self._m:
            bins = self._m
        #
        from numpy import floor,array,matrix,r_,zeros,linalg,sum,abs,where,eye
        #
        #
        nrsperlst = int(floor(self._m/bins))
        nrx = self._m-nrsperlst
        
        Cov = matrix([self.f(r,epsilon) for r in self.radii.flatten()]).reshape(-1,self._k)
        
        
        for i in range(bins):
            
            lstskp1,lstskp2 = i*nrsperlst,(i+1)*nrsperlst
            randlstuse = r_[self.randlst[:lstskp1],self.randlst[lstskp2:]]
            randlstcheck = self.randlst[lstskp1:lstskp2]
            
            if self._m == self._k:
                PHI = Cov[randlstuse][:,randlstuse]
                f_k = matrix(self.f_i)[randlstuse]
            else:
                PHI = Cov[randlstuse].T*Cov[randlstuse]
                f_k = Cov[randlstuse].T*matrix(self.f_i)[randlstuse]
                
            PHI = self._set_smoothing(PHI,iprint=iprint)
            
            w_k = array(linalg.solve(PHI,f_k)).flatten()
            
            for j in randlstcheck:
                if self._m == self._k:
                    dists = self.radii[j][randlstuse]
                else:
                    dists = self.radii[j]
                
                rads = array([self.f(r,epsilon) for r in dists])
                
                comps = w_k*rads
                curr_err = abs(sum(comps[where((comps>0)|(comps<0))])-self.f_i[j])[0]
                
                self.f_xve[j] = curr_err
        
        return sum(self.f_xve)
    
    
        
    
    
    
    
    
    
    
    def log_likelihood(self,epsilon=None,iprint=0):
        '''
        
    >>  Calculate the logarithmic likelihood function for a specific value of
        [ epsilon ] 
    
    
    >>  Possible keyword arguments and default values:
        
        epsilon : float
            The kernel function length scale
            If None, the current value is evaluated
        
        '''
        
        
        if epsilon is None:
            epsilon = self.epsilon
            
        from numpy import abs,matrix,linalg,ones,isnan,isinf,sqrt,pi,exp,log,eye,sign
        from numpy.linalg import cond
        
        epsilon = abs(epsilon)
        
        n = self._k
        
        Cov = matrix([self.f(r,epsilon) for r in self.radii.flatten()]).reshape(-1,self._k)
           
           
        if not self._k == self._m :  # set up [PHI]{w_k} = {f_k}  to reprensent least squares regression
            
            PHI = Cov.T*matrix(Cov)
            f_k = Cov.T*matrix(self.f_i)
            
            PHI = self._set_smoothing(PHI,iprint=iprint)
            
            U,S,V = linalg.svd(PHI, full_matrices=True)
                
            Sinv = eye(S.size)/sqrt(S)
                
            PHIinv = matmul(V.T,matmul(Sinv,U.T))
            PHIdet = sum(sqrt(S))
            
            f_k = matmul(self.PHIinv,f_k)
            
        else: # solve directly
            
            PHI = matrix(Cov)
            f_k = matrix(self.f_i)
                         
            PHI = self._set_smoothing(PHI,iprint=iprint)
            
            PHIinv = linalg.inv(PHI)
            PHIdet = linalg.det(PHI)
        
        #
        if isnan(PHIdet):
            Phidet = 1e-10
        if isinf(PHIdet):
            Phidet = 1e10
        if PHIdet<1e-10:
            PHIdet = 1e-10
            
            
        P_f_k  = PHIinv*f_k
        onesv = matrix(ones(n)).T
        oPo = onesv.T*PHIinv*onesv
        
        # f_mean
        f_mu = (onesv.T*PHIinv*f_k/oPo)[0,0]
        # modified 
        f_mod = f_k - f_mu
        # weights
        w_k = PHIinv*f_mod
        # check the weighting for NaNs and INFs
        w_k[isnan(w_k)]=0.
        w_k[isinf(w_k)]=0.
        # f_s2
        f_s2 = (f_mod.T*w_k)[0,0]/n
        
        
        if isnan(f_s2):
            f_s2 = 1e-10
        if isinf(f_s2):
            f_s2 = 1e10
        if f_s2<1e-10:
            f_s2 = 1e-10
        
        ML0 = 1./(sqrt(PHIdet)*(2*pi*f_s2)**(n/2))
        ML1 = exp(-n/2.)
        
        LLE = log(ML0*ML1)
        
        if isinf(LLE):
            LLE = sign(LLE)*1e10
            
        if isnan(LLE):
            LLE = 0
    
        return LLE
    
    
    
    
    
    
    
    
    def fit(self,using='likelihood',bins=None,iprint=0,opt='golden'):
        '''
        
    >>  Fit the kernel length scale parameter
        [ epsilon ] 
    
    
    >>  Possible keyword arguments and default values:
        
        using : string
            Fit by either maximising the log likelihood function
            i.e. using = 'likelihood'
            OR
            by minimising the cross-validation error
            i.e. using = 'cross-validate'
            
        bins : integer
            Only if:
            using = 'cross-validate'
            The number of random cross-vaildation sample bins used to calculate
            the cross-validation error
        
        '''
        
        if using=='cross-validate':
            # minimise the cross validation error
            f_obj = lambda x : self.cross_validate(abs(self.epsilon+x),bins,iprint=iprint)
        else:
            # maximise the log likelihood function
            f_obj = lambda x : -self.log_likelihood(abs(self.epsilon+x),iprint=iprint)
        
        from scipy.optimize import fmin_slsqp,golden
        
        
        if opt=='golden':
            xopt = golden(f_obj)
        else:
            xopt = fmin_slsqp(f_obj,0,iprint=iprint)
        
        self.kwdict['epsilon'] = abs(self.epsilon+xopt)
        
        # set the RBF functions
        self._set_funct()
        #
        # calculate the weights
        self._calc_weights()
    
        
     
      
      
      
      
      
            
            
    def predict(self,x,**kwargs):
        
        '''
            
    >>  Return an approximate to the the function value using the RBF 
        surrogate model
                         
         f(x) ~  S(x) =  SUM   [  beta  x phi (x)  + mu  ]
                            k         k      k
                            
    
    Parameters:
    
        x : array_like
            Input variable prediction array of shape [M]x[N]
            i.e. 
            return [M] predictions of the approximated [N] dimensional space  
            
    >>  Example:
        
        
        f = lambda x: sin(8*x) + 0.66*cos(18*x) + 3 - 2*x
        
        x_i = linspace(0,1,5)
        f_i = f(x_i)
        
        
        f_rbf = surrogate.rbf(x_i,f_i)
        
        x = array([0.1,0.2,0.8]).reshape(-1,1)
        
        f_predicted = f_rbf.predict(x)
    
    
        
        '''
        return self.__call__(x,False,False)
         
         
         
         
     
     
      
        
    def predict_gradient(self,x):
        
        '''
            
    >>  Return an approximate function gradient using the RBF 
        surrogate model
                         
        d[f(x)]/dx  ~  d [S(x)]/dx  =  SUM   [  beta  x grad (phi (x)  ]
                                          k         k           k
                               
    
    Parameters:
    
        x : array_like
            Input variable prediction array of shape [M]x[N]
            i.e. 
            return [M] prediction of the approximated [N] dimensional space  
            
    >>  Example:
        
        
        f = lambda x: sin(8*x) + 0.66*cos(18*x) + 3 - 2*x
        
        x_i = linspace(0,1,5)
        f_i = f(x_i)
        
        
        f_rbf = surrogate.rbf(x_i,f_i)
        
        x = array([0.1,0.2,0.8]).reshape(-1,1)
        
        dfdx_predicted = f_rbf.predict_gradient(x)
    
    
        
        '''
    
    
    
    
        Xnew = Xnew.reshape(-1,self.k)
        DYnew = []
        for xx in Xnew:
            xx = xx.flatten() - self.avg
            #print("XX = ",xx)
            eps = self.epsilon
            xdist = self.sf*(self.xi - xx)
            dists = sqrt(sum(xdist*xdist,1))
            #print(dists)
            ddxlst = array([-xi/dists for xi in xdist.T])
            #print(ddxlst)
            
            drads = self.df(dists,eps)
            #print(drads)
            
            if not self.multi:
                #print('')
                comps = array([array(self.weights*drads*dx).flatten() for dx in ddxlst]).T
                comps[isnan(comps)]=0.
                comps[isinf(comps)]=0.
                dy = sum(comps,0)
                #print('  dy :  ',dy)
    
            else:
                dyarr = []
                for i in range(self.nry):
                    comps = self.weights[i]*drads*ddx
                    dyarr+=[sum(comps[where((comps>0)|(comps<0))])]
                dy = array(dyarr)
            
            
            DYnew+=[dy]
            
        return array(DYnew)
        
            
            
     
     
     
     
     
    
    
#    
#    
#    
#    def optimize_EPS(self,method='slsqp'):
#        
#        x0 = self.epsilon
#        xopt = opt.fmin(self.validationEPS,x0)
#        
#        self.epsilon = xopt[0]
#    
#    
#
#    def validationSFS(self,sf=1.,q=5):
#        '''
#        Dimensional scaling factor??
#        '''
#        
#        if array(sf).size < self.k:
#            sfs = ones(self.k)*array([sf]).flatten()[0]
#        else:
#            sfs = array([sf]).flatten()[:self.k]
#        nrsperlst = int(floor(self.n/q))
#        nrx = self.n-nrsperlst
#        validERR = 0.
#        for i in range(q):
#            lstskp1,lstskp2 = i*nrsperlst,(i+1)*nrsperlst
#            randlstuse = r_[self.randlst[:lstskp1],self.randlst[lstskp2:]]
#            randlstcheck = self.randlst[lstskp1:lstskp2]
#            mat = zeros((nrx,nrx))
#            for ii in range(nrx):
#                inrx = randlstuse[ii]
#                xdist = sfs*(self.xi[randlstuse] - self.xi[inrx])
#                mat[ii,:] = self.f(sqrt(sum(xdist*xdist,1)))
#                mat[ii,ii] = self.cor0
#            matinv = linalg.inv(mat)
#            if not self.multi:
#                mu =  (matrix(ones(nrx))*matinv*matrix(self.yi[randlstuse])/(matrix(ones(nrx))*matinv*matrix(ones(nrx)).T))[0,0]
#                weights =  linalg.solve(mat,self.yi[randlstuse]-mu)
#            else:
#                mu,sdev2,weights = {},{},{}
#                for i in range(self.nry):
#                    mu[i] =  (matrix(ones(nrx))*matinv*matrix(self.yi[randlstuse,i]).T/(matrix(ones(nrx))*matinv*matrix(ones(nrx)).T))[0,0]
#                    weights[i] =  linalg.solve(mat,self.yi[randlstuse,i]-mu[i])
#            for j in randlstcheck:
#                xdist = sfs*(self.xi[randlstuse] - self.xi[j])
#                dists = sqrt(sum(xdist*xdist,1))
#                rads = self.f(dists)
#                if not self.multi:
#                    comps = weights*rads
#                    ynew = sum(comps[where((comps>0)|(comps<0))])+mu
#                    validERR += abs((ynew-self.yi[j]))#/self.yi[j])
#                else:
#                    ynew = []
#                    for ii in range(self.nry):
#                        comps = weights[ii]*rads
#                        ynew+=[sum(comps[where((comps>0)|(comps<0))])+mu[ii]]
#                    ynew = array(ynew)
#                    validERR += sum(abs((ynew-self.yi[j])))#/self.yi[j]))
#        return validERR
#      
#      
#      
#    def MSE(self,xnew,epsilon=None):
#        if epsilon is None:
#            epsilon = self.epsilon
#        xnew = array([xnew]).flatten()-self.avg
#        oneV = matrix(ones(self.n)).T
#        ri = oneV*(1-self.cor0)
#        xdist = self.sf*(self.xi - xnew)
#        #xdist = self.xi - xnew
#        dists = sqrt(sum(xdist*xdist,1))
#        riC = matrix(self.f(dists,epsilon)).T
#        ri[array(riC>0).flatten()] = riC[array(riC>0).flatten()]
#        ri[array(riC<0).flatten()] = riC[array(riC<0).flatten()]
#        #
#        # from DR Jones Schonlau and Welch
#        s2 = self.sdev2*(self.cor0-ri.T*self.matinv*ri+(self.cor0-oneV.T*self.matinv*ri)**2/(oneV.T*self.matinv*oneV))[0,0]
#        if isnan(s2):
#            s2 = 0.
#        return max([s2,0.])
#    
#    
#    
#    
#    def cumProb(self,xi,toler=1.e-5):# calculate probability that y <= fmin
#        mean = self(xi)
#        std = sqrt(max([self.MSE(xi),0]))
#        PV = norm(mean,std).cdf(min(self.yi))
#        if isnan(PV):
#            PV=0.
#        return PV
#    
#
#    def Likelihood(self,xi,toler=1.e-5):
#        y0 = min(self.yi)
#        mean = self(xi)
#        std = sqrt(max([self.MSE(xi),0]))
#        A = exp(-(y0-mean)**2/std)
#        if std == 0:
#            A = 0
#        if isnan(A):
#            A = 0.
#        return A
#    
#    
#    def EI(self,xi,intrule=10):
#        fmin = min(self.yi)
#        mean = self(xi)
#        std = sqrt(max([self.MSE(xi),0]))
#    
#        term1 = 1./(std*sqrt(2*pi))
#        term2 = -0.5/std**2
#        p = lambda x: term1*exp(term2*(x-mean)**2)
#        PHIx = 0.
#        f0 = mean - 10*std
#        if f0 < fmin:
#            h = (fmin-f0)/(2.*intrule)
#            xlst = r_[f0,f0+(array(range(2*intrule))+1)*h]
#            plst = p(xlst)
#            PHIx = plst[0]+plst[-1]
#            if intrule>1:
#                lst2 = 2*(array(range(intrule-1))+1)
#                PHIx +=2*sum(plst[lst2])
#            lst4 = 2*(array(range((intrule)))+1)-1
#            PHIx +=4*sum(plst[lst4])
#            PHIx = PHIx*h/3
#        EIval = PHIx*(fmin-mean)+std*p(fmin)
#        if isnan(EIval):
#            EIval = 0.
#        
#        return max([EIval,0]),std*p(fmin)
#    
#    
#    
#    def GEI(self,xi,g=3,useabs=False):
#        g = int(g)
#        fmin = min(self.yi)
#        mean = self(xi)
#        std = sqrt(max([self.MSE(xi),0]))
#        fnmin = (fmin-mean)/std
#        nf = norm(mean,std)
#        CV = nf.cdf(fmin)
#        PV = nf.pdf(fmin)
#        TkC = CV
#        TkN = -CV
#        EIg = 0.
#        if useabs:
#            fnminabs = abs(fnmin)
#            for k in range(g+1):
#                EIg += (std**g)*comb(g,k)*(fnminabs**(g-k))*TkC
#                TkP = TkC
#                TkC = TkN
#                TkN =  PV*fnminabs**(k+1)+(k+1)*TkP
#            return EIg
#        # Else the following is executed
#        for k in range(g+1):
#            EIg += (std**g)*((-1)**k)*comb(g,k)*(fnmin**(g-k))*TkC
#            TkP = TkC
#            TkC = TkN
#            TkN =  -PV*fnmin**(k+1)+(k+1)*TkP
#        
#        if isnan(EIg):
#            EIg = 0.
#        
#        return max([EIg,0.])
#    
#  
    