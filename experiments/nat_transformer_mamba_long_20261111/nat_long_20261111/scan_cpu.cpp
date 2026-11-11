// CPU contiguous affine recurrence, O(B*T*I*N), no OpenMP or fast math.
#include <cstdint>
template<class F> void forward(const F* a,const F* b,F* h,int64_t B,int64_t T,int64_t D){
 for(int64_t batch=0;batch<B;++batch){
  const int64_t base=batch*T*D;
  for(int64_t j=0;j<D;++j) h[base+j]=b[base+j];
  for(int64_t t=1;t<T;++t){
   const int64_t p=base+t*D;
   for(int64_t j=0;j<D;++j) h[p+j]=a[p+j]*h[p-D+j]+b[p+j];
  }
 }
}
template<class F> void backward(const F* a,const F* h,const F* g,F* ga,F* gb,int64_t B,int64_t T,int64_t D){
 for(int64_t batch=0;batch<B;++batch){
  const int64_t base=batch*T*D;
  for(int64_t t=T-1;t>=0;--t){
   const int64_t p=base+t*D;
   if(t==T-1){for(int64_t j=0;j<D;++j) gb[p+j]=g[p+j];}
   else {for(int64_t j=0;j<D;++j) gb[p+j]=g[p+j]+a[p+D+j]*gb[p+D+j];}
   if(t==0){for(int64_t j=0;j<D;++j) ga[p+j]=F(0);}
   else {for(int64_t j=0;j<D;++j) ga[p+j]=gb[p+j]*h[p-D+j];}
  }
 }
}
extern "C" {
void affine_forward_f32(const float* a,const float* b,float* h,int64_t B,int64_t T,int64_t D){forward(a,b,h,B,T,D);}
void affine_backward_f32(const float* a,const float* h,const float* g,float* ga,float* gb,int64_t B,int64_t T,int64_t D){backward(a,h,g,ga,gb,B,T,D);}
void affine_forward_f64(const double* a,const double* b,double* h,int64_t B,int64_t T,int64_t D){forward(a,b,h,B,T,D);}
void affine_backward_f64(const double* a,const double* h,const double* g,double* ga,double* gb,int64_t B,int64_t T,int64_t D){backward(a,h,g,ga,gb,B,T,D);}
}
