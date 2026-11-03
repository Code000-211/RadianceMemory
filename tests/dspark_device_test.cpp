/* Focused device/CPU DSpark differential. Build with hipcc, then pass libr4d.so.
 * This intentionally calls the real exported launchers, not the host SIMT emulator.
 * See docs/DSPARK-VALIDATION-WIP.md for a scoped invocation and its limitations. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <vector>

using Fn = int (*)(const RadArgs*, RadStream);
static void check(bool b, const char* s) { if (!b) { std::fprintf(stderr, "FAIL %s\n", s); std::exit(1); } }
static void hipcheck(hipError_t e) { if (e != hipSuccess) { std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(e)); std::exit(1); } }
static uint16_t bf(float f) { uint32_t x; std::memcpy(&x, &f, 4); x += 0x7fff + ((x >> 16) & 1); return x >> 16; }
static float fb(uint16_t b) { uint32_t x = (uint32_t)b << 16; float f; std::memcpy(&f, &x, 4); return f; }
template<class T> struct Device {
    T* p = nullptr; size_t n;
    explicit Device(const std::vector<T>& v): n(v.size()) { hipcheck(hipMalloc(&p, n*sizeof(T))); hipcheck(hipMemcpy(p, v.data(), n*sizeof(T), hipMemcpyHostToDevice)); }
    ~Device() { (void)hipFree(p); }
    std::vector<T> get() { std::vector<T> v(n); hipcheck(hipMemcpy(v.data(), p, n*sizeof(T), hipMemcpyDeviceToHost)); return v; }
};
static RadTensor t(void* p, int dt, std::initializer_list<int64_t> sh) {
    RadTensor x{}; x.data=p; x.dtype=dt; x.rank=sh.size(); int i=0; for (auto d:sh) x.shape[i++]=d; rad_tensor_pack(&x); return x;
}
static RadParam pi(const char* k, long long v) { RadParam p{}; p.key=k; p.kind=RAD_P_INT; p.ival=v; return p; }
static RadParam ps(const char* k, const char* v) { RadParam p{}; p.key=k; p.kind=RAD_P_STR; p.sval=v; return p; }
static Fn symbol(void* h, const char* name) { auto f = reinterpret_cast<Fn>(dlsym(h,name)); check(f != nullptr, name); return f; }

static void markov(Fn fn, int R, bool f32) {
    constexpr int M=2, S=2, V=130, T=3, LP=137, SP=5, TP=7, OP=4;
    std::vector<uint16_t> l(M*S*LP,bf(0)), p(V*R,bf(0)), u(V*R,bf(0));
    for (int m=0;m<M;++m) for(int v=0;v<16;++v) l[(m*S)*LP+v]=bf(100-v);
    p[1*R]=bf(1); u[127*R]=bf(200); u[129*R]=bf(200);
    p[127*R+1]=bf(1); u[100*R+1]=bf(100);
    p[2*R+2]=bf(1); u[80*R+2]=bf(200);
    p[80*R+3]=bf(1); u[101*R+3]=bf(90);
    std::vector<int> anchor{1,-1,2,-1}, ti(M*TP,-777), out(M*OP,-777);
    std::vector<float> ts(M*SP,-777.f);
    std::vector<float> lf(l.size()); for(size_t i=0;i<l.size();++i) lf[i]=fb(l[i]);
    Device<uint16_t> dl(l), dp(p), du(u); Device<int> da(anchor), di(ti), dout(out);
    Device<float> ds(ts), dlf(lf);
    RadTensor a[]={t(f32 ? (void*)dlf.p : (void*)dl.p,f32 ? RAD_F32 : RAD_BF16,{M*S,V}),t(da.p,RAD_I32,{M,2}),t(dp.p,RAD_BF16,{V,R}),t(du.p,RAD_BF16,{V,R}),t(ds.p,RAD_F32,{M,T}),t(di.p,RAD_I32,{M,T}),t(dout.p,RAD_I32,{M,S})};
    a[0].stride[0]=LP; a[4].stride[0]=SP; a[5].stride[0]=TP; a[6].stride[0]=OP;
    RadParam par[]={pi("M",M),pi("steps",S),pi("rank",R),pi("n_vocab",V),pi("anchor_stride",2),ps("dtype","bf16")};
    RadArgs args{}; args.t=a; args.n_t=7; args.p=par; args.n_p=6;
    check(fn(&args,nullptr)==RAD_OK,"markov launch"); hipcheck(hipDeviceSynchronize());
    auto go=dout.get(); auto gs=ds.get(); auto gi=di.get();
    // Independent scalar oracle. Values are deliberately exactly representable so reduction order
    // cannot explain a discrepancy. Sequential conditioning and tile winners are both checked.
    for(int m=0;m<M;++m) { int prev=anchor[m*2];
        for(int s=0;s<S;++s) { float best=-INFINITY; int winner=0;
            for(int v=0;v<V;++v) { float score=fb(l[(m*S+s)*LP+v]);
                for(int r=0;r<R;++r) score+=fb(p[prev*R+r])*fb(u[v*R+r]);
                if(score>best) { best=score; winner=v; }
            }
            check(go[m*OP+s]==winner,"markov CPU winner"); prev=winner;
        }
        for(int tile=0;tile<T;++tile) { float best=-INFINITY; int win=tile*64; int prev_final=go[m*OP];
            for(int v=tile*64;v<std::min(V,(tile+1)*64);++v) { float score=fb(l[(m*S+1)*LP+v]);
                for(int r=0;r<R;++r) score+=fb(p[prev_final*R+r])*fb(u[v*R+r]);
                if(score>best) {best=score; win=v;}
            }
            check(gs[m*SP+tile]==best && gi[m*TP+tile]==win,"markov pitched tile oracle");
        }
        for(int i=T;i<SP;++i) check(gs[m*SP+i]==-777.f,"score row padding guard");
        for(int i=T;i<TP;++i) check(gi[m*TP+i]==-777,"token row padding guard");
        for(int i=S;i<OP;++i) check(go[m*OP+i]==-777,"output padding guard");
    }
    check(go[0]==127 && go[1]==100 && go[OP]==80 && go[OP+1]==101,"top16/tie/sequential controls");
    a[4].stride[1]=2; check(fn(&args,nullptr)!=RAD_OK,"reject non-unit scratch inner stride"); a[4].stride[1]=1;
    a[5].stride[0]=T-1; check(fn(&args,nullptr)!=RAD_OK,"reject overlapping scratch rows");
    a[5].stride[0]=LLONG_MAX; check(fn(&args,nullptr)!=RAD_OK,"reject overflowing scratch span");
    std::printf("markov logits=%s rank=%d padded scores=%d tokens=%d: PASS\n",f32?"f32":"bf16",R,SP,TP);
}

// A controlled precision counterexample, not a measurement of real checkpoint logits.
static void near_tie(Fn fn, Fn fn_f32) {
    constexpr int R=32, V=2;
    std::vector<uint16_t> l{bf(1.001f),bf(1.002f)}, w(V*R,bf(0));
    std::vector<int> anchor{0}, token{-1}, tile{-1}; std::vector<float> score{-1};
    check(l[0]==l[1],"near tie quantizes identically");
    Device<uint16_t> dl(l), dw(w); Device<int> da(anchor), dt(tile), dout(token); Device<float> ds(score);
    RadTensor a[]={t(dl.p,RAD_BF16,{1,V}),t(da.p,RAD_I32,{1}),t(dw.p,RAD_BF16,{V,R}),t(dw.p,RAD_BF16,{V,R}),t(ds.p,RAD_F32,{1,1}),t(dt.p,RAD_I32,{1,1}),t(dout.p,RAD_I32,{1,1})};
    RadParam par[]={pi("steps",1),pi("rank",R),pi("n_vocab",V),pi("anchor_stride",1)};
    RadArgs args{}; args.t=a; args.n_t=7; args.p=par; args.n_p=4;
    check(fn(&args,nullptr)==RAD_OK,"near tie launch"); hipcheck(hipDeviceSynchronize());
    check(dout.get()[0]==0 && 1.002f>1.001f,"BF16/F32 near tie divergence");
    std::vector<float> full{1.001f,1.002f}; Device<float> df(full);
    a[0]=t(df.p,RAD_F32,{1,V});
    check(fn(&args,nullptr)!=RAD_OK,"BF16 op refuses F32 logits");
    check(fn_f32(&args,nullptr)==RAD_OK,"F32 near tie launch"); hipcheck(hipDeviceSynchronize());
    check(dout.get()[0]==1,"F32 near tie retains winner");
    a[0]=t(dl.p,RAD_BF16,{1,V});
    check(fn_f32(&args,nullptr)!=RAD_OK,"F32 op refuses BF16 logits");
    std::puts("near tie: F32 argmax=1, BF16 input argmax=0: PASS (synthetic)");
}

// Real device head -> F32 Markov chain, with an exactly representable BF16-input near tie.
static void head_chain(Fn head, Fn walk) {
    constexpr int K=8, V=2, R=32;
    std::vector<uint16_t> x(K,bf(0)), w(V*K,bf(0)), cb(V*R,bf(0));
    x[0]=bf(1); x[1]=bf(0.001953125f); w[0]=bf(1); w[K]=bf(1); w[K+1]=bf(1);
    std::vector<float> logits(V,-1), scores(1,-1);
    std::vector<int> anchor{0}, tile{-1}, out{-1};
    Device<uint16_t> dx(x), dw(w), dc(cb); Device<float> dl(logits), ds(scores);
    Device<int> da(anchor), dt(tile), dout(out);
    RadTensor h[]={t(dx.p,RAD_BF16,{1,K}),t(dw.p,RAD_BF16,{V,K}),t(dl.p,RAD_F32,{1,V})};
    RadParam hp[]={pi("M",1),pi("n_vocab",V),pi("n_embd",K),ps("dtype","bf16")};
    RadArgs ha{}; ha.t=h; ha.n_t=3; ha.p=hp; ha.n_p=4;
    check(head(&ha,nullptr)==RAD_OK,"F32 logits head launch"); hipcheck(hipDeviceSynchronize());
    auto got=dl.get(); check(got[0]==1.f && got[1]==1.001953125f,"F32 logits head CPU dot oracle");
    check(bf(got[0])==bf(got[1]),"head-chain BF16 negative control");
    RadTensor a[]={t(dl.p,RAD_F32,{1,V}),t(da.p,RAD_I32,{1}),t(dc.p,RAD_BF16,{V,R}),t(dc.p,RAD_BF16,{V,R}),t(ds.p,RAD_F32,{1,1}),t(dt.p,RAD_I32,{1,1}),t(dout.p,RAD_I32,{1,1})};
    RadParam par[]={pi("M",1),pi("steps",1),pi("rank",R),pi("n_vocab",V),ps("dtype","bf16")};
    RadArgs args{}; args.t=a; args.n_t=7; args.p=par; args.n_p=5;
    check(walk(&args,nullptr)==RAD_OK,"head-chain walk launch"); hipcheck(hipDeviceSynchronize());
    check(dout.get()[0]==1,"F32 head-chain winner");
    std::puts("F32 head -> Markov near tie and BF16 negative control: PASS");
}

static void attention(Fn shared, Fn ordinary, int G, int ctx0, int ctx1, int causal) {
    constexpr int M=2, Q=7, H=256, BS=16, W=513;
    const int ctx[M]={ctx0,ctx1}, blocks=(std::max(ctx0,ctx1)+15)/16, nblocks=M*blocks;
    std::vector<uint16_t> q(M*Q*G*H,bf(0)), kv((size_t)nblocks*BS*2*H,bf(0)), out(M*Q*G*H,bf(0));
    std::vector<int> bt(M*blocks), su{ctx0,ctx1}, cu{0,Q,2*Q};
    // Reverse physical pages: the oracle must consult block_table rather than logical positions.
    for(int s=0;s<M;++s) for(int b=0;b<blocks;++b) bt[s*blocks+b]=nblocks-1-(s*blocks+b);
    for(int s=0;s<M;++s) for(int j=0;j<ctx[s];++j) {
        size_t base=((size_t)bt[s*blocks+j/BS]*BS+j%BS)*2*H;
        kv[base]=bf((j%5-2)*0.25f);
        float v=(j%7-3)*0.25f;
        int anchor=ctx[s]-Q;
        if(j==anchor-W) v=512;
        if(j==anchor-W+1) v=64;
        if(j==ctx[s]-1) v=256;
        kv[base+H]=bf(v);
    }
    for(int row=0;row<M*Q;++row) for(int h=0;h<G;++h) q[((row*G+h)*H)]=bf(h%2 ? -1.f : 1.f);
    Device<uint16_t> dq(q), dk(kv), dout(out); Device<int> db(bt), ds(su), dc(cu);
    RadTensor a[]={t(dq.p,RAD_BF16,{M*Q,G,H}),t(dk.p,RAD_BF16,{nblocks,1,BS,2*H}),
        t(db.p,RAD_I32,{M,blocks}),t(ds.p,RAD_I32,{M}),RadTensor{},RadTensor{},
        t(dc.p,RAD_I32,{M+1}),t(dout.p,RAD_BF16,{M*Q,G,H})};
    RadParam par[]={pi("q_len",Q),pi("head_dim",H),pi("gqa",G),pi("block_size",BS),
        pi("causal",causal),pi("window",W),pi("n_head",G),pi("max_seqs",M),
        ps("q_dtype","bf16"),ps("kv_dtype","bf16")};
    RadArgs args{}; args.t=a; args.n_t=8; args.p=par; args.n_p=10;
    check(shared(&args,nullptr)==RAD_OK,"shared anchor attention launch"); hipcheck(hipDeviceSynchronize());
    auto got=dout.get(); std::vector<double> expected(M*Q*G);
    for(int s=0;s<M;++s) for(int row=0;row<Q;++row) for(int h=0;h<G;++h) {
        int anchor=ctx[s]-Q, lo=std::max(0,anchor-W+1), hi=causal?anchor+1:ctx[s];
        double sum=0, acc=0;
        for(int j=lo;j<hi;++j) {
            size_t base=((size_t)bt[s*blocks+j/BS]*BS+j%BS)*2*H;
            double weight=std::exp((h%2?-1.:1.)*fb(kv[base])/16.0);
            sum+=weight; acc+=weight*fb(kv[base+H]);
        }
        double want=acc/sum; size_t i=(s*Q+row)*G+h;
        expected[i]=want;
        check(std::abs(fb(got[i*H])-want)<0.012,"attention scalar oracle");
        check(fb(got[i*H+1])==0,"attention zero channel");
    }
    // The ordinary per-query window must fail this fixture; this guards against a vacuous test.
    check(ordinary(&args,nullptr)==RAD_OK,"ordinary attention negative control launch");
    hipcheck(hipDeviceSynchronize()); auto wrong=dout.get(); bool distinguished=false;
    for(size_t i=0;i<expected.size();++i)
        if(std::abs(fb(wrong[i*H])-expected[i])>0.03) distinguished=true;
    check(distinguished,"attention per-row window negative control");
    std::printf("attention gqa=%d ctx=%d,%d causal=%d shared oracle and negative control: PASS\n",G,ctx0,ctx1,causal);
}

// Real checkpoint head slice, but synthetic BF16 activations and zero Markov weights. This tests
// head arithmetic, not full-graph/reference equivalence or real hidden-state acceptance.
static void checkpoint_head(Fn head, Fn walk, const char* weights_path) {
    constexpr int M=16, V=1025, K=2560, R=32, T=(V+63)/64;
    std::vector<uint16_t> w((size_t)V*K), x((size_t)M*K), cb((size_t)V*R,bf(0));
    std::ifstream f(weights_path,std::ios::binary); check((bool)f,"open checkpoint head slice");
    f.read(reinterpret_cast<char*>(w.data()),w.size()*sizeof(uint16_t));
    check((size_t)f.gcount()==w.size()*sizeof(uint16_t) && f.peek()==EOF,"exact head slice size");
    uint32_t rng=123456789; for(auto& v:x) { rng=rng*1664525u+1013904223u; v=bf(((int)(rng>>24)-128)/128.f); }
    std::vector<float> logits((size_t)M*V,0), scores((size_t)M*T,0);
    std::vector<int> anchor(M,0), tile((size_t)M*T,-1), out(M,-1);
    Device<uint16_t> dx(x), dw(w), dc(cb); Device<float> dl(logits), ds(scores);
    Device<int> da(anchor), dt(tile), dout(out);
    RadTensor h[]={t(dx.p,RAD_BF16,{M,K}),t(dw.p,RAD_BF16,{V,K}),t(dl.p,RAD_F32,{M,V})};
    RadParam hp[]={pi("M",M),pi("n_vocab",V),pi("n_embd",K),ps("dtype","bf16")};
    RadArgs ha{}; ha.t=h; ha.n_t=3; ha.p=hp; ha.n_p=4;
    check(head(&ha,nullptr)==RAD_OK,"checkpoint head launch"); hipcheck(hipDeviceSynchronize());
    auto got=dl.get(); double maxerr=0, se=0, sr=0, minmargin=INFINITY; int changed=0,ties=0;
    std::vector<int> refwin(M);
    for(int m=0;m<M;++m) { double best=-INFINITY, second=-INFINITY; int winner=0;
        for(int v=0;v<V;++v) { double ref=0; for(int k=0;k<K;++k) ref+=(double)fb(x[m*K+k])*fb(w[(size_t)v*K+k]);
            double err=std::abs(got[(size_t)m*V+v]-ref); maxerr=std::max(maxerr,err); se+=err*err; sr+=ref*ref;
            check(std::isfinite(got[(size_t)m*V+v]),"finite checkpoint head logits");
            if(ref>best) {second=best;best=ref;winner=v;} else if(ref>second) second=ref;
        }
        refwin[m]=winner; minmargin=std::min(minmargin,best-second);
        int gw=0,bw=0,nmax=0; float gmax=-INFINITY, bmax=-INFINITY;
        for(int v=0;v<V;++v) {float y=got[(size_t)m*V+v], by=fb(bf(y));
            if(y>gmax) {gmax=y;gw=v;} if(by>bmax) {bmax=by;bw=v;nmax=1;} else if(by==bmax) ++nmax;
        }
        if(gw!=bw) ++changed; if(nmax>1) ++ties;
        check(gw==winner,"checkpoint head F32 top1 versus FP64 oracle");
    }
    RadTensor a[]={t(dl.p,RAD_F32,{M,V}),t(da.p,RAD_I32,{M}),t(dc.p,RAD_BF16,{V,R}),t(dc.p,RAD_BF16,{V,R}),t(ds.p,RAD_F32,{M,T}),t(dt.p,RAD_I32,{M,T}),t(dout.p,RAD_I32,{M,1})};
    RadParam p[]={pi("M",M),pi("steps",1),pi("rank",R),pi("n_vocab",V),ps("dtype","bf16")};
    RadArgs args{};args.t=a;args.n_t=7;args.p=p;args.n_p=5;
    check(walk(&args,nullptr)==RAD_OK,"checkpoint head walk"); hipcheck(hipDeviceSynchronize());
    auto picks=dout.get(); for(int m=0;m<M;++m) check(picks[m]==refwin[m],"checkpoint F32 chain top1");
    std::printf("real head slice V=%d K=%d synthetic rows=%d: max_abs=%.9g rel_l2=%.9g min_top_margin=%.9g F32_vs_FP64_top1_mismatch=0 BF16_rounded_top1_changed=%d BF16_top_ties=%d\n",V,K,M,maxerr,std::sqrt(se/sr),minmargin,changed,ties);
}

int main(int argc,char** argv) {
    check(argc==2 || (argc==4 && !std::strcmp(argv[2],"--head-weights")),"usage: dspark_device_test libr4d.so [--head-weights file]");
    int n_devices = 0;
    if (hipGetDeviceCount(&n_devices) != hipSuccess || n_devices == 0) return 77;
    void* h=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    if(!h) { std::fprintf(stderr,"dlopen: %s\n",dlerror()); return 1; }
    Fn fn=symbol(h,"r4d_rad_dflash_markov_greedy");
    Fn fn_f32=symbol(h,"r4d_rad_dflash_markov_greedy_f32");
    if(argc==4) { checkpoint_head(symbol(h,"r4d_rad_logits_gemm"),fn_f32,argv[3]); dlclose(h); return 0; }
    for(int r:{32,64,128,512}) { markov(fn,r,false); markov(fn_f32,r,true); }
    near_tie(fn,fn_f32);
    head_chain(symbol(h,"r4d_rad_logits_gemm"),fn_f32);
    for(int g:{1,2}) {
        const char* shared=g==1?"r4d_rad_attn_anchor_window_h256_gqa1_bf16kv":"r4d_rad_attn_anchor_window_h256_gqa2_bf16kv";
        const char* ordinary=g==1?"r4d_rad_attn_window_h256_gqa1_bf16":"r4d_rad_attn_window_h256_gqa2_bf16";
        for(int causal:{0,1}) {
            attention(symbol(h,shared),symbol(h,ordinary),g,520,1049,causal);
            attention(symbol(h,shared),symbol(h,ordinary),g,521,1050,causal);
        }
    }
    dlclose(h); std::puts("DSpark device differential: PASS");
}
