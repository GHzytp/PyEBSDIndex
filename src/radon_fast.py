import numpy as np
from timeit import default_timer as timer
from os import path
from numba import jit, prange
import pyopencl as cl
import matplotlib.pyplot as plt
from os import environ
environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

RADDEG = 180.0/np.pi
DEGRAD = np.pi/180.0

class Radon():
  def __init__(self, image=None, imageDim = None, nTheta = 180, nRho=90,rhoMax = None):
    self.nTheta = nTheta
    self.nRho = nRho
    self.rhoMax = rhoMax
    self.indexPlan = None
    if (image is None) and (imageDim is None):
      self.theta = None
      self.rho = None
      self.imDim = None
    else:
      if (image is not None):
        self.imDim = np.asarray(image.shape[-2:])
      else:
        self.imDim = np.asarray(imageDim[-2:])
      self.radon_plan_setup(imageDim=self.imDim, nTheta=self.nTheta, nRho=self.nRho, rhoMax=self.rhoMax)

  def radon_plan_setup(self, image=None, imageDim=None, nTheta=None, nRho=None, rhoMax=None):
    if (image is None) and (imageDim is not None):
      imDim = np.asarray(imageDim, dtype=np.int)
    elif (image is not None):
      imDim =  np.shape(image)[-2:] # this will catch if someone sends in a [1 x N x M] image
    else:
      return -1
    imDim = np.asarray(imDim)
    self.imDim = imDim
    if (nTheta is not None) : self.nTheta = nTheta
    if (nRho is not None): self.nRho = nRho
    self.rhoMax = rhoMax if (rhoMax is not None) else np.round(np.linalg.norm(imDim)*0.5)

    deltaRho = float(2 * self.rhoMax) / (self.nRho)
    self.theta = np.arange(self.nTheta, dtype = np.float32)*180.0/self.nTheta
    self.rho = np.arange(self.nRho, dtype = np.float32)*deltaRho - (self.rhoMax-deltaRho)

    #xmin = -1.0*(self.imDim[0]-1)*0.5
    #ymin = -1.0*(self.imDim[1]-1)*0.5
    xmin = -1.0*(self.imDim[0]-1)*0.5
    ymin = -1.0*(self.imDim[1]-1)*0.5

    #self.radon = np.zeros([self.nRho, self.nTheta])
    sTheta = np.sin(self.theta*DEGRAD)
    cTheta = np.cos(self.theta*DEGRAD)
    thetatest = np.abs(sTheta) >= (np.sqrt(2.) * 0.5)

    m = np.arange(self.imDim[0], dtype = np.uint32)
    n = np.arange(self.imDim[1], dtype = np.uint32)

    a = -1.0*np.where(thetatest == 1, cTheta, sTheta)
    a /= np.where(thetatest == 1, sTheta, cTheta)
    b = xmin*cTheta + ymin*sTheta

    self.indexPlan = np.zeros([self.nRho, self.nTheta, self.imDim.max()], dtype=np.uint64)
    outofbounds = self.imDim[0]*self.imDim[1]
    for i in np.arange(self.nTheta):
      b1 = self.rho - b[i]
      if thetatest[i]:
        b1 /= sTheta[i]
        b1 = b1.reshape(self.nRho, 1)
        indx_y = np.floor(a[i]*m+b1).astype(np.int64)
        indx_y = np.where(indx_y < 0, outofbounds, indx_y)
        indx_y = np.where(indx_y >= self.imDim[1], outofbounds, indx_y)
        #indx_y = np.clip(indx_y, 0, self.imDim[1])
        indx1D = np.clip(m+self.imDim[0]*indx_y, 0, outofbounds)
        self.indexPlan[:,i, :] = indx1D
      else:
        b1 /= cTheta[i]
        b1 = b1.reshape(self.nRho, 1)
        if cTheta[i] > 0:
          indx_x = np.floor(a[i]*n + b1).astype(np.int64)
        else:
          indx_x = np.ceil(a[i] * n + b1).astype(np.int64)
        indx_x = np.where(indx_x < 0, outofbounds, indx_x)
        indx_x = np.where(indx_x >= self.imDim[0], outofbounds, indx_x)
        indx1D = np.clip(indx_x+self.imDim[0]*n, 0, outofbounds)
        self.indexPlan[:, i, :] = indx1D
      self.indexPlan.sort(axis = -1)


  def radon_fast(self, image, padding = np.array([0,0]), fixArtifacts = False):
    tic = timer()
    shapeIm = np.shape(image)
    if image.ndim == 2:
      nIm = 1
      image = image[np.newaxis, : ,:]
      reform = True
    else:
      nIm = shapeIm[0]
      reform = False

    nPx = shapeIm[-1]*shapeIm[-2]
    im = np.zeros(nPx+1, dtype=np.float32)
    #radon = np.zeros([nIm, self.nRho, self.nTheta], dtype=np.float32)
    radon = np.zeros([nIm,self.nRho + 2 * padding[0],self.nTheta + 2 * padding[1]],dtype=np.float32)
    shpRdn = radon.shape
    norm = np.sum(self.indexPlan < nPx, axis = 2 ) + 1.0e-12
    for i in np.arange(nIm):
      im[:-1] = image[i,:,:].flatten()
      radon[i, padding[0]:shpRdn[1]-padding[0], padding[1]:shpRdn[2]-padding[1]] = np.sum(im.take(self.indexPlan.astype(np.int64)), axis=2) / norm

    if (fixArtifacts == True):
      radon[:,:,0] = radon[:,:,1]
      radon[:,:,-1] = radon[:,:,-2]

    radon = np.transpose(radon, [1,2,0]).copy()

    if reform==True:
      image = image.reshape(shapeIm)

    #print(timer()-tic)
    return radon

  def radon_faster(self,image,padding = np.array([0,0]), fixArtifacts = False):
    tic = timer()
    shapeIm = np.shape(image)
    if image.ndim == 2:
      nIm = 1
      #image = image[np.newaxis, : ,:]
      #reform = True
    else:
      nIm = shapeIm[0]
    #  reform = False

    image = image.reshape(-1)

    nPx = shapeIm[-1]*shapeIm[-2]
    indxDim = np.asarray(self.indexPlan.shape)
    #radon = np.zeros([nIm, self.nRho+2*padding[0], self.nTheta+2*padding[1]], dtype=np.float32)
    radon = np.zeros([self.nRho + 2 * padding[0],self.nTheta + 2 * padding[1], nIm],dtype=np.float32)
    shp = radon.shape

    self.rdn_loops(image,self.indexPlan,nIm,nPx,indxDim,radon, np.asarray(padding))

    if (fixArtifacts == True):
      radon[:,padding[1],:] = radon[:,padding[1]+1,:]
      radon[:,shp[1]-1-padding[1],:] = radon[:,shp[1]-padding[1]-2,:]


    image = image.reshape(shapeIm)

    #print(timer()-tic)
    return radon

  @staticmethod
  @jit(nopython=True, fastmath=True, cache=True, parallel=False)
  def rdn_loops(images,index,nIm,nPx,indxdim,radon, padding):
    nRho = indxdim[0]
    nTheta = indxdim[1]
    nIndex = indxdim[2]
    count = 0.0
    sum = 0.0
    for q in prange(nIm):
      imstart = q*nPx
      for i in range(nRho):
        ip = i+padding[0]
        for j in range(nTheta):
          jp = j+padding[1]
          count = 0.0
          sum = 0.0
          for k in range(nIndex):
            indx1 = index[i,j,k]
            if (indx1 >= nPx):
              break
            #radon[q, i, j] += images[imstart+indx1]
            sum += images[imstart + indx1]
            count += 1.0
          radon[ip,jp,q] = sum/(count + 1.0e-12)

  def radon_fasterCL(self,image,padding = np.array([0,0]),fixArtifacts = False, returnBuff = False, clparams=[None, None, None, None, None] ):

    tic = timer()
    if isinstance(clparams[1],cl.Context):
      gpu = clparams[0]
      ctx = clparams[1]
      prg = clparams[3]
      if isinstance(clparams[2], cl.CommandQueue):
        queue = clparams[2]
      else:
        queue = cl.CommandQueue(ctx)
      mf = clparams[4]
    else:
      try:
        gpu = cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)
        ctx = cl.Context(devices={gpu[0]})
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags
        kernel_location = path.dirname(__file__)
        prg = cl.Program(ctx,open(path.join(kernel_location,'clkernels.cl')).read()).build()
      except:
        return self.radon_faster(image,padding=padding,fixArtifacts = fixArtifacts)

    shapeIm = np.shape(image)
    if image.ndim == 2:
      nIm = 1
      image = image.reshape(1, shapeIm[0], shapeIm[1])
      shapeIm = np.shape(image)
    else:
      nIm = shapeIm[0]
    #  reform = False

    clvtypesize = 16 # this is the vector size to be used in the openCL implementation.
    nImCL = np.int32(clvtypesize * (np.int(np.ceil(nIm/clvtypesize))))
    # there is something very strange that happens if the number of images
    # is a exact multiple of the max group size (typically 256)
    mxGroupSz = gpu[0].get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
    nImCL += np.int(16 * (1 - np.int(np.mod(nImCL, mxGroupSz ) > 0)))
    image_align = np.zeros((shapeIm[1], shapeIm[2], nImCL), dtype = np.float32)
    image_align[:,:,0:nIm] = np.transpose(image, [1,2,0]).copy()
    radon = np.zeros([self.nRho+2*padding[0],self.nTheta+2*padding[1], nImCL],dtype=np.float32)
    radon_gpu = cl.Buffer(ctx,mf.READ_WRITE,size=radon.nbytes)
    #radon_gpu = cl.Buffer(ctx,mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf=radon)
    image_gpu = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=image_align)
    rdnIndx_gpu = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=self.indexPlan)

    imstep = np.uint64(np.product(shapeIm[-2:]))
    indxstep = np.uint64(self.indexPlan.shape[-1])
    rdnstep = np.uint64(self.nRho * self.nTheta)
    shpRdn = np.asarray(radon.shape, dtype = np.uint64)
    padRho = np.uint64(padding[0])
    padTheta = np.uint64(padding[1])
    tic = timer()

    nImChunk = np.uint64(nImCL/clvtypesize)
    prg.radonSum(queue,(nImChunk,rdnstep),None,rdnIndx_gpu,image_gpu,radon_gpu,
                  imstep, indxstep,
                 shpRdn[0], shpRdn[1],
                 padRho, padTheta, np.uint64(self.nTheta))


    if (fixArtifacts == True):
       prg.radonFixArt(queue,(nImChunk,self.nRho),None,radon_gpu,
                       shpRdn[0],shpRdn[1],padTheta)

    #image = image.reshape(shapeIm)
    queue.flush()
    rdnIndx_gpu.release()
    image_gpu.release()
    if returnBuff == False:
      cl.enqueue_copy(queue,radon,radon_gpu,is_blocking=True).wait()
      radon_gpu.release()
      radon = radon[:,:, 0:nIm]
      radon_gpu = None
      clparams = [None, None, None, None, None]
      return radon, clparams, radon_gpu
    else:
      clparams = [gpu,ctx,queue,prg,mf]
      return radon, clparams, radon_gpu

    #if (fixArtifacts == True):
    #  radon[:,:,padding[1]] = radon[:,:,padding[1]+1]
    #  radon[:,:,-padding[1]-1] = radon[:,:,-2-padding[1]]




    #print(timer()-tic)





# if __name__ == "__main__":
#   import ebsd_pattern, ebsd_index
#   file = '~/Desktop/SLMtest/scan2v3nlparl09sw7.up1' ;f = ebsd_pattern.UPFile(file)
#
#   pat = f.read_data(patStartEnd=[0,1],convertToFloat=True,returnArrayOnly=True )
#   dat, indxer = ebsd_index.index_pats(filename = file, patStart = 0, patEnd = 1,return_indexer_obj = True)
#   dat = ebsd_index.index_pats_distributed(filename = file,patStart = 0, patEnd = -1, chunksize = 1000, ncpu = 34, ebsd_indexer_obj = indxer )
#