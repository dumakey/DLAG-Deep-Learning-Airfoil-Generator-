import numpy as np
from scipy.interpolate import interpolate
import math
import re
import os
from collections import OrderedDict
import functions

'''
Script that reads an airfoil coordinates file (x,z) and computes the following 
design parameters:
*Relative thickness
*Maximum thickness x-coordinate
*Leading edge radious
*[dz/dx]_max
*[dz/dx]_min

'''

class AirfoilScanner():
	def __init__(self,coordinates_filepath,parameters):
		self.filepath = coordinates_filepath
		self.parameter_function_launcher = {
				'leradius':self.set_LEradius,
				'teangle':self.set_TEangle,
				'tmax':self.set_relative_thickness,
				'zmax':self.set_zmax,
				'zmin':self.set_zmin,
				'dzdx':self.set_curvature_controlpoints,
				}
		self.parameters = parameters
		'''
		self.parameters = dict.fromkeys(parameters['parameters'])
		if len(parameters) != 0:
			if parameters['dzdx_cp'] != None:
				if 'dzdx_c' in parameters['parameters']:
					del self.parameters['dzdx_c']
					self.parameters['dzdx'] = ('camber',parameters['dzdx_cp'])
				elif 'dzdx_t' in parameters['parameters']:
					del self.parameters['dzdx_t']
					self.parameters['dzdx'] = ('thickness',parameters['dzdx_cp'])
		'''

	def refine(self, x, z):
		xq = []
		zq = []
		for i in range(x.__len__()):
			xq.append(np.concatenate([self.sinspace(x[i][0],0.5,-int(0.75*x[i].size)),
									  self.sinspace(0.501,x[i][-1],int(0.75*x[i].size))]))
			zq.append(self.interp1d(x[i],z[i],xq[-1],'quadratic'))

		return xq, zq

	def interp1d(self,x,y,xq,method):

		try:
			f = interpolate.interp1d(x,y,method)
			yq = f(xq)
		except ValueError:
			f = interpolate.interp1d(x,y,fill_value='extrapolate')
			yq = f(xq)

		return np.float64(yq)

	def sinspace(self,d1,d2,n):

		if n < 0:
			N = -n
			y = d2*np.ones(N) + (d1 - d2) * np.ones(N) * list(map(lambda x:math.sin(x),math.pi/2 * (1 - (np.arange(0,N,1))/np.float((N - 1)))))
		else:
			y = d1 * np.ones(n) + (d2 - d1) * np.ones(n) * list(map(lambda x: math.sin(x),math.pi/(2*(n - 1)) * np.arange(0,n,1)))


		'''
		if factor:
			y = (1 - factor) * y + factor * [d1*np.ones(n) + np.arange(0,n-1,1) * (d2 - d1)*np.ones(n)/np.float(n - 1), d2*np.ones(n)]
	
		'''
		return y

	def refine_interpolation_domains(self,x,z,k,*args):
		xref = self.sinspace(x[0], x[-1], k*len(x))
		#xref = np.linspace(x[0],x[-1], k * len(x))
		if args:
			zref = self.interp1d(x,z,xref,args[0])
		else:
			zref = self.interp1d(x,z,xref,'linear')

		return xref, zref

	def get_geometry(self):

		name = self.filepath.split(os.sep)[-1].replace('.dat','')
		#print('Airfoil: ',name)

		### Get geometry
		with open(self.filepath, 'r') as f:
			geo = f.readlines()
			f.close()

		# Locate upper and lower side indexes
		iUS = 0
		iLS = [index for index in range(len(geo)) if 'LS' in geo[index]][0]

		### Get upper side
		xu = []
		zu = []
		for line in geo[iUS+1:iLS]:
			xu.append(np.float(line.split()[0]))
			zu.append(np.float(line.split()[1]))
		xu, zu = self.refine([np.array(xu)],[np.array(zu)])
		xu = list(xu[0])
		zu = list(zu[0])

		### Get lower side
		xl = []
		zl = []
		for line in geo[iLS + 1:]:
			xl.append(np.float(line.split()[0]))
			zl.append(np.float(line.split()[1]))
		xl, zl = self.refine([np.array(xl)],[np.array(zl)])
		xl = list(xl[0])
		zl = list(zl[0])

		### Conversion to numpy arrays
		xu = np.array(xu)
		zu = np.array(zu)
		xl = np.array(xl)
		zl = np.array(zl)

		### Get camber line
		int_method = 'quadratic'
		x = np.linspace(0,1,int(1.3*max(xu.size,xl.size)))
		zc = 0.5 * (self.interp1d(xu,zu,x,int_method) + self.interp1d(xl,zl,x,int_method))
		# Get thickness distribution
		zt = 0.5 * (self.interp1d(xu,zu,x,int_method) - self.interp1d(xl,zl,x,int_method))

		# Refine curvature
		_, zc_r = self.refine([x],[zc])
		x_r, zt_r = self.refine([x],[zt])
		x = x_r[0]
		zc = zc_r[0]
		zt = zt_r[0]

		# Both arrays must begin at the LE and end at the TE. Flip in case the raw data is not set this way
		if xu[1] - xu[0] < 0:
			xu = np.flipud(xu)
			zu = np.flipud(zu)
		if xl[1] - xl[0] < 0:
			xl = np.flipud(xl)
			zl = np.flipud(zl)

		# Store upper and lower side geometry for further calculations
		self.xup = (name,xu)
		self.zup = (name,zu)
		self.xlow = (name,xl)
		self.zlow = (name,zl)
		self.x = (name,x)
		self.zc = (name,zc)
		self.zt = (name,zt)

		return xu, zu, xl, zl, x, zc, zt, name

	def set_LEradius(self, args):

		xu, zu, xl, zl = args
		# Define limit x-coordinate
		xc = xu[3]

		for i in range(xu.size):
			if xu[i] > xc:
				ic_u = i
				break
		for i in range(xl.size):
			if xl[i] > xc:
				ic_l = i
				break

		# Construct x and z coordinates range
		x = np.concatenate((np.flipud(xu[0:ic_u]), xl[0:ic_l]))
		z = np.concatenate((-np.flipud(zu[0:ic_u]), -zl[0:ic_l]))

		_, idx = np.unique(x, return_index=True)
		x = x[np.sort(idx)]
		z = z[np.sort(idx)]

		z,x = self.refine_interpolation_domains(z,x,10,'quadratic')

		# Approximate x,z to quadratic function
		[a,b,_] = np.polyfit(z,x,2)

		# Compute LE z-coordinate
		z_bar = z[np.where(x == min(x))]

		# Compute approximate radius
		r_bar = 1/(2*abs(a)) * (4*a*z_bar * (a*z_bar + b) + b**2 + 1)**1.5

		self.design_parameters['leradius'] = r_bar[0]

	def set_TEangle(self, args):

		xu, zu, xl, zl = args

		dzudx = np.gradient(zu,xu)
		dzldx = np.gradient(zl,xl)

		#Thetau = math.degrees(math.atan(dzudx[-1]))
		#Thetal = math.degrees(math.atan(dzldx[-1]))

		Thetau = math.atan(dzudx[-1])
		Thetal = math.atan(dzldx[-1])

		self.design_parameters['teangle'] = abs(Thetau - Thetal)

	def set_zmax(self, args):

		xu, zu = args

		self.design_parameters['zmax'] = max(zu)
		self.design_parameters['xzmax'] = xu[np.where(zu == self.design_parameters['zmax'])][0]

	def set_zmin(self, args):

		xl, zl = args

		self.design_parameters['zmin'] = min(zl)
		self.design_parameters['xzmin'] = xl[np.where(zl == self.design_parameters['zmin'])][0]

	def set_curvature_controlpoints(self, args):

		feature = args[0]
		if feature == 'camber':
			y = self.zc[1]
			x = self.x[1]
		elif feature == 'thickness':
			y = self.zt[1]
			x = self.x[1]

		np.seterr(all='ignore')  # Command to ignore possible "division by zero" warnings
		# Compute first derivative of y w.r.t. x
		dydx = np.gradient(y,x)
		# Filter possible "NaN" or "inf" values
		idx = [i for i in range(len(x)) if math.isinf(dydx[i]) == False and math.isnan(dydx[i]) == False]
		x = x[idx]
		dydx = dydx[idx]

		xc = args[1]
		dzdx =  self.interp1d(x,dydx,xc,'linear')

		ii = 1
		for item in dzdx:
			if math.isnan(item) == False:
				self.design_parameters['dzdx_%s_%s'%(feature,ii)] = item
				ii += 1

	def set_relative_thickness(self, args):

		xu, zu, xl, zl = args

		# Define interpolation range
		x = np.linspace(xu[0],xu[-1],xu.size)

		# Interpolate upper side
		zupper = self.interp1d(xu,zu,x,'linear')
		# Interpolate lower side
		zlower = self.interp1d(xl,zl,x,'linear')

		dz = zupper-zlower
		tmax = max(dz)
		c = max(xu) - min(xu)

		self.design_parameters['xtmax'] = x[np.where(dz == tmax)[0][0]]
		self.design_parameters['tmax'] = round(tmax/c,3)

	def set_parameters(self):

		self.design_parameters = OrderedDict()
		for parameterID in self.parameters.keys():
			if parameterID == 'leradius':
				fun_args = (self.xup[1], self.zup[1], self.xlow[1], self.zlow[1])
			elif parameterID == 'teangle':
				fun_args = (self.xup[1], self.zup[1], self.xlow[1], self.zlow[1])
			elif parameterID == 'tmax':
				fun_args = (self.xup[1], self.zup[1], self.xlow[1], self.zlow[1])
			elif parameterID == 'zmax':
				fun_args = (self.xup[1], self.zup[1])
			elif parameterID == 'zmin':
				fun_args = (self.xlow[1], self.zlow[1])
			elif parameterID == 'dzdx':
				controlpoints = self.parameters['dzdx']
				fun_args = (controlpoints)

			self.parameter_function_launcher[parameterID](fun_args)

	@staticmethod
	def get_internal_parameters(z, x, m):
		r = x.__len__()
		a = np.zeros([m+1, ])
		g_inv = 1/functions.class_function()(x,1,0.5,z[0],z[1])
		ii = 1
		g_inv_max = g_inv[ii:-ii].max()
		while g_inv_max > 40:
			ii += 1
			g_inv_max = g_inv[ii:-ii].max()

		z_div_g = np.multiply(g_inv[ii:-ii], z[ii:-ii].reshape(g_inv.size - 2 * ii, 1))
		P = functions.shape_functions()(x,m)
		D = np.linalg.pinv(P[:, ii:-ii].T)
		# aeqv = np.reshape(D * z_div_g,(m + 1,))
		# zest = np.multiply(1/g_inv,P.T * aeqv.T)
		# a = np.reshape(D * z_div_g,(m + 1,))
		a = np.array(D * z_div_g)

		return a

	@staticmethod
	def reconstruct_z(z, x, order=6, provide_a=False, a=None):

		if provide_a == False:
			g_inv = 1/self.class_function(x,1,0.5,z[0],z[-1])
			ii = 1
			g_inv_max = g_inv[ii:-ii].max()
			while g_inv_max > 40:
				ii += 1
				g_inv_max = g_inv[ii:-ii].max()

			z_div_g = np.multiply(g_inv[ii:-ii],z[ii:-ii].reshape(g_inv.size - 2 * ii,1))
			P = self.shape_functions(x,order)
			D = np.linalg.pinv(P[:,ii:-ii].T)
			a = np.reshape(D * z_div_g,(order + 1,1))
		else:
			P = functions.shape_functions()(x,order)

		g = functions.class_function()(x,1,0.5,z[0],z[-1])

		return np.squeeze(np.array(np.multiply(g, P.T * a)))

def get_aerodata(parameters, case_folder, airfoil_analysis, add_geometry=False, fmt='dat'):

	plots_folder = os.path.join(case_folder,'Datasets','geometry','originals')
	if airfoil_analysis == 'camber':
		airfoil_fpaths = [os.path.join(plots_folder,file) for file in os.listdir(plots_folder)
						  if file.endswith(fmt) if not file.endswith('_s%s' % fmt)]
	elif airfoil_analysis == 'thickness':
		airfoil_fpaths = [os.path.join(plots_folder,file) for file in os.listdir(plots_folder) if file.endswith(fmt)]

	airfoil_data = dict()
	for fpath in airfoil_fpaths:
		airfoil_scanner = AirfoilScanner(fpath,parameters)
		xu, zu, xl, zl, x, zc, zt, name = airfoil_scanner.get_geometry()
		airfoil_scanner.set_parameters()
		airfoil_data[name] = {}
		if add_geometry == True:
			airfoil_data[name] = {'x':x,'zc':zc,'zt':zt,'xu':xu,'zu':zu,'xl':xl,'zl':zl}
		airfoil_data[name].update(airfoil_scanner.design_parameters)

	return airfoil_data

