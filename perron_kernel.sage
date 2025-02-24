# supplementary code for the paper: 
# "The least balanced graphs and trees" by Péter Csikvári, Viktor Harangi
# link:  https://arxiv.org/abs/2502.13939

import numpy as np
from numpy.linalg import eigh,inv
from sage.graphs.trees import TreeIterator

Ga_conn=0.5*(3*sqrt(3.)+5)  # ~5.098 
Ga_tree=4+2*sqrt(3.)        # ~7.4641 
q=lambda x: x/2-sqrt((x/2)^2-1)
r=lambda x: x/2+sqrt((x/2)^2-1)
t = polygen(ZZ,'t')
la_var = polygen(ZZ,'\u03BB')
ka_var = polygen(ZZ,'\u03BA')

def principal_ev(A):
    evals,evecs=eigh(A)
    ind=np.argmax(evals)
    la=evals[ind]
    vec=evecs[:,ind]
    return la,vec if vec[0]>=0 else -vec

def lambda_G(G):
    return principal_ev(G.adjacency_matrix())[0]

def Gamma(vec):
    return sum(vec)^2/sum(vec^2)

def GammaA(A):
    return Gamma(principal_ev(A)[1])

def GammaG(G):
    return GammaA(G.adjacency_matrix())

def draw_graph(G,colored_vertices=[],title=None):
    G.graphplot(title=title,graph_border=(title!=None),layout='spring', iterations=1000,vertex_colors={'lightblue':colored_vertices}).show(figsize=2+0.5*G.diameter())
        
def possible_root(G,v,tree_version=False):
    if G.degree(v)<3:
        return False
    if tree_version:
        return True
    for u in G.vertices():
        if G.degree(u)>G.degree(v) and all(u==w or G.has_edge(u,w) for w in G.neighbors(v)):
            return False
    return True

# generate the possible n-kernels
# (rooted conn.graphs/trees on n vertices such that the root has degree >= 3 and is not "dominated" by another vertex)
def generate_kernels(n,only_trees=False):
    rooted_graphs=[]
    it=TreeIterator(n) if only_trees else graphs.nauty_geng('-c {}'.format(n))
    for G in it:
        orbs=G.automorphism_group().orbits()
        for orb in orbs:
            root=orb[0]
            if possible_root(G,root,only_trees):
                rooted_graphs.append((G,root))
    print("{} rooted {} found on {} vertices".format(len(rooted_graphs),"trees" if only_trees else "graphs",n))
    return rooted_graphs

#def V_act(G,root,use_lvl0_extension=False):
#    levels=[[] for _ in range(4)]
#    for v in G.vertices():
#        levels[G.distance(root,v)].append(v)        
#    if use_lvl0_extension:
#        levels[0]+=[v for v in levels[1] if all(u==v or G.has_edge(u,v) for u in levels[1])]
#        levels[1]=[v for v in levels[1] if v not in levels[0]]
#    return levels[3]+levels[2] if len(levels[3])>0 else levels[2]+levels[1]

def power_set(S,item_excl=None,exclude_empty=True):
    subsets = [[]]
    for item in S:
        subsets += [ss+[item] for ss in subsets]
    if item_excl!=None:
        i=S.index(item_excl)
        subsets.pop(2^i)
    if exclude_empty:
        subsets.pop(0)
    return subsets

def glue(G0,v,G1,u=0):
    n=G0.order()
    G1.relabel({i:n+i for i in range(G1.order())})
    G=G0.union(G1)
    G.add_edge(v,n+u)
    return G

def unifork(n):
    G=graphs.PathGraph(n-1)
    add_leaf(G,1)
    return G

def bifork(n):
    G=graphs.PathGraph(n-2)
    add_leaf(G,1)
    add_leaf(G,n-4)
    return G

def add_path(G,k,v=None):
    n=G.order()
    if v==None:
        v=n-1
    G.add_edge(v,n)
    for i in range(k-1):
        G.add_edge(n+i,n+i+1)

def add_leaf(G,v=None):
    add_path(G,1,v)

# our bound on Gamma in terms of the degree of the master vertex
def beta_deg(d,show_plot=False):
    sqd=sqrt(d+0.)
    la_deg=find_root((x+1)^3-(d+1)*(3*x+1),sqd,2*sqd)
    fun=lambda x: x/(1-(d+1)/(x+1)^2)
    if show_plot:
        show( point((la_deg,fun(la_deg)),color='red')+plot(fun,sqd,2*sqd) )
    return fun(la_deg)

beta_d_map={i: beta_deg(i) for i in range(3,21)}

# sort n-vertex conn.graphs/trees in a descending order of their Gamma
# displays the first display_nr graphs
def ranking(n,only_trees=False,display_nr=10):
    it=TreeIterator(n) if only_trees else graphs.nauty_geng('-c {}'.format(n))
    Gas=[(G,GammaG(G)) for G in it]
    Gas_sorted=sorted(Gas, key=lambda pair: pair[1])

    if display_nr>0:
        str="trees" if only_trees else "connected graphs"
        print("{}-vertex {} with the smallest Gamma:".format(n,str))
        for pair in Gas_sorted[:display_nr]:
            draw_graph(pair[0])
            print(pair[1])
        print("---------------------------------------------------------")

    return Gas_sorted

# locates a root of the function 'fun' that should be positive at 'start' 
# (useful when the root may be very close to 'start') 
def find_r(fun,start,step=1./16,prec=0.5^20):
    assert fun(start)>0
    while fun(start+step)>=0:
        start+=step
    end=start+step
    while end-start>prec:
        mid=0.5*(start+end)
        if fun(mid)>=0:
            start=mid
        else:
            end=mid
    return start,end

# checks whether the coefficients of a (translated) polynomial are all positive
def check_coeffs(pol,t0=0):
    trans_pol=pol if t0==0 else pol(t+t0)
    return all(co>=0 for co in trans_pol.coefficients())

# for a polynomial pol, checks whether pol>0 on the interval (t0,t1)    
def is_positive_pol(pol,t0,t1):
    tt=0
    if t0 == -np.inf:
        tt= 0 if t1 == np.inf else t1-1        
    else:    
        tt= t0+1 if t1 == np.inf else 0.5*(t0+t1)
    if pol(tt)<=0:
        return False
    polR=0.+pol
    rts=polR.roots()
    for rt in rts:
        if rt[0]>t0 and rt[0]<t1 and rt[1]%2:
            return False
    return True

# checks whether a rational function is positive on (t0,t1)
def is_positive(rat,t0,t1):
    f=rat.numerator()
    g=rat.denominator()
    if f(0.5*(t0+t1))<0:
        f=-f
        g=-g
    return is_positive_pol(f,t0,t1) and is_positive_pol(g,t0,t1)

# checks whether a rational function is monotone increasing on (t0,t1)
def is_monotone(rat,t0,t1):
    f=rat.numerator()
    g=rat.denominator()
    pol=f.derivative()*g-f*g.derivative()
    return is_positive_pol(pol,t0,t1)

def true_minimum(num,denum,t0,delta=2.):
    fun=lambda x: num(x)/denum(x)
    val0=fun(t0)
    diff_pol=num-val0*denum
    while all(co>=0 for co in diff_pol(t+t0+delta).coefficients()):
        delta/=2
    t1=t0+2*delta
    val=find_local_minimum(fun,t0,t1)[0]
    return min(val,fun(t0),fun(t1))


class Kernel:
    def __init__(self, H, root=0, Ga_target=0,tree_mode=False):
        self.H=H.copy()
        self.root=root
        self.tree_mode=tree_mode
        self.Ga_target=Ga_target # target ratio (beta)
        
        if self.Ga_target<=0:
            self.Ga_target=Ga_tree if self.tree_mode else Ga_conn

        self.n=self.H.order() # nr of vertices
        self.A=self.H.adjacency_matrix() # adjacency matrix
        mat=la_var*matrix.identity(self.n)-self.A # lambda*I-A
        self.P=mat.det() # char-poly of A
        self.adj=mat.adjugate() # adjoint(tI-A)=P(tI-A)^{-1}=PB
        self.la_H,self.xx=principal_ev(self.A) # top eigval and principal eigvec
        self.Ga_H=Gamma(self.xx) # Gamma_H = Gamma(x)
        
        self.active=list(range(self.n)) # set of active vertices
        
        self.A_ext=np.zeros((self.n+1,self.n+1)) # an auxiliary matrix of size (n+1)x(n+1)
        self.A_ext[:-1,:-1]=self.A
        

    def draw(self,with_root=True,title=None):
        draw_graph(self.H,[self.root] if with_root else [],title)
    
    def la_U(self,U,Gamma_version=False):
        row=np.zeros(self.n+1)
        row[U]=1
        self.A_ext[-1]=row
        self.A_ext[:,-1]=row
        la,vec=principal_ev(self.A_ext)
        if Gamma_version:
            return Gamma(vec)
        else:
            return la
    
    def default_active(self):
        dists=[self.H.distance(self.root,v) for v in range(self.n)]
        dist_max=max(dists)
        self.active=[v for v in range(self.n) if dists[v]>=dist_max-1]
        if self.root in self.active:
            d=self.H.degree(self.root)
            #print(d,beta_d_map[d+1],self.Ga_target)
            if beta_d_map[d+1]>self.Ga_target:
                self.active.remove(self.root)  # we may remove root from active b/c Gamma > beta_{d+1} > Ga_target 
                
    # setting Uc=\mathcal(U) based on the set of active vertices
    # Uc = set of nonempty subsets of active, excluding the one-element set {v_excl} unless v_excl=None 
    # in tree-mode: Uc=one-element subsets of active
    def act_Uc(self,v_excl=None,update=True):
        self.Uc=[[v] for v in self.active] if self.tree_mode else power_set(self.active,v_excl)
        if update:
            self.update_Uc()            
    
    # setting your own Uc (not based on active)
    def set_Uc(self,Uc):
        self.Uc=Uc
        self.update_Uc()
            
    def update_Uc(self):
        self.nr=len(self.Uc) # nr of sets in Uc
        self.las=[0.]*self.nr # list of la_U's for U in Uc
        self.Bt=[None]*self.nr # essentially $P \tilde(B)_{v,U}$ for U in Uc
        self.s=[None]*self.nr # essentially $P s_U$ for U in Uc
        for i in range(self.nr):
            U=self.Uc[i]
            self.las[i]=self.la_U(U)
            self.Bt[i]=sum(self.adj[u] for u in U)
            self.s[i]=sum(self.Bt[i])
    
    # remove sets with given indices from Uc and update las,Bt,s accordingly
    # indices should be in increasing order
    def remove_from_Uc(self,indices):
        for i in reversed(indices):
            self.Uc.pop(i)
            self.las.pop(i)
            self.Bt.pop(i)
            self.s.pop(i)
        self.nr=len(self.Uc)
            
    # returns lambda and Gamma of H +_v P_infty (infinite path attached to H at v) 
    def ev_infty(self,v):
        if self.la_H<2:
            if self.n>8:
                ngh=self.H.neighbors(v)
                if len(ngh)==1 and self.H.degree(ngh[0])==2:
                    return 2.,np.inf  # in this case H must be a path or "unifork" with v being the endpoint
            K_plus=self.augment_kernel([v])
            return K_plus.ev_infty(self.n)
        la_infty=find_root(self.adj[v,v](x)*q(x)==self.P(x),self.la_H,self.n)
        bb=self.adj[v](la_infty)/self.P(la_infty)
        qq=q(la_infty)
        Ga_infty=(sum(bb)+1/(1-qq))^2/(sum(val^2 for val in bb)+1/(1-qq^2))
        return la_infty,Ga_infty

    # checks if Ga_infty <= Ga_target for some active v
    def infty_check(self):
        below_target=False
        for v in self.active:
            la_infty,Ga_infty=self.ev_infty(v)
            if Ga_infty<=self.Ga_target:
                below_target=True
                txt="infinite FAMILY: \u0393\u2192 {:.5f} (through vertex {})".format(Ga_infty,v)
                draw_graph(self.H,[v],txt)
        return below_target
    
    # checks condition for bound
    # if returns True, then every Uc-extension of the kernel (H,root) must have Ga_G >= Ga_target
    # if 'diagonal_only' is set to 'True': check the condition only for pairs U=V and saves the indices of failed U's 
    def simple_check(self,diagonal_only=False):
        self.check_fails=[]
        for i1 in range(self.nr):
            i2_bnd=i1+1 if diagonal_only else self.nr
            for i2 in range(i1,i2_bnd):
                lat=max(self.las[i1],self.las[i2]) # \tilde{\lambda}
                cUV=sum(self.Bt[i1][v]*self.Bt[i2][v] for v in range(self.n))
                num=(self.s[i1]+self.P)*(self.s[i2]+self.P)
                denum=cUV+(self.Bt[i1][self.root]+self.Bt[i2][self.root])*self.P/2
                if not check_coeffs(num-self.Ga_target*denum,lat):
                    if diagonal_only:
                        self.check_fails.append(i1)
                    else:    
                        return False
        return len(self.check_fails)==0    

    # returns the augmented/extended kernel obtained by adding a new vertex to H and connecting to 'nghbrs'
    def augment_kernel(self,nghbrs):
        H_plus=self.H.copy()
        for v in nghbrs:
            H_plus.add_edge(v,self.n)
        K_plus=Kernel(H_plus,self.root,self.Ga_target,self.tree_mode)
        K_plus.active=self.active+[self.n]
        K_plus.act_Uc()
        return K_plus
                
    # if returns True, then {v} may be removed from Uc (or v can be removed from active in tree-mode)
    def rule_out_singleton(self,v):
        K_plus=self.augment_kernel([v])
        if K_plus.Ga_H<=self.Ga_target:
            txt="{}-vertex GRAPH: \u0393={:.5f}".format(K_plus.n,K_plus.Ga_H)
            K_plus.draw(False,txt)
        return K_plus.simple_check() 
    
    def two_step_check(self):
        ind=np.argmin([self.la_U([v]) for v in self.active])
        v_min=self.active[ind] # the "minimal" active vertex, for which la_{v} is the smallest 
        print("two-step-check excluding #{} vertex started...".format(v_min))
        step1=self.rule_out_singleton(v_min)
        self.act_Uc(v_min)    
        step2=self.simple_check()
        if step1 and step2:
            print("SUCCESSFUL")
        else:
            print("FAILED!!! step 1:{}  step 2: {}".format(step1,step2))

    # active-vertex-elimination process: checks active vertices one by one to see if they can be removed
    def active_elimination(self):
        assert self.tree_mode
        V=self.active.copy()
        for v in V:
            if self.rule_out_singleton(v):
                self.active.remove(v)

    # if, after active-vertex-elimination, only one active vertex v remains active,
    # then we can continue with an augmented kernel (extended through v)  
    def augment_check(self,n_limit=16):
        if self.n>n_limit:
            print("!!!augment-check FAILED: vertex nr LIMIT EXCEEDED ({})".format(n_limit))
            return False
        if self.simple_check():
            return True
        print("simple-check with active set {} FAILED for:".format(self.active))
        self.draw()
        print("active-vertex-elimination started...")
        self.active_elimination()
        if len(self.active)==0:
            print("...no active vertex remained!")
            print("augment-check SUCCESSFUL")
            return True
        elif len(self.active)==1:
            v=self.active[0]
            print("...only one active vertex remained --> adding leaf at vertex #{}...".format(v))
            K_plus=self.augment_kernel([v])
            return K_plus.augment_check(n_limit)
        else:
            print(" ...multiple active vertices remained:",self.active)
            print("!!!augment-check FAILED")
            return False            

    # searches for families and single graphs (extending the kernel) with Gamma_G< Ga_target
    # by recording which subsets U fail the simple-check, then 
    # for every failed U, an augmented kernel is considered (with one new vertex connected to the vertices in U)
    # search() is recursively called for each augmented kernel
    def search(self,n_limit):
        if self.n>n_limit:
            print("!!! vertex number LIMIT EXCEEDED ({})".format(n_limit))
            return
        #print("uc started", self.n)
        if self.infty_check():
            return
        self.simple_check(True)
        U_failed=[self.Uc[i] for i in self.check_fails]
        self.remove_from_Uc(self.check_fails)
        assert self.simple_check()  # the remaining Uc should pass this test (unless "non-diagonal" pairs U!=V cause problems)
        #print("failed sets:",U_failed)
        #print("remaining:",self.Uc)
        for U in U_failed:
            K_plus=self.augment_kernel(U)
            if K_plus.Ga_H<=self.Ga_target:
                txt="{}-vertex GRAPH: \u0393={:.5f}".format(K_plus.n,K_plus.Ga_H)
                K_plus.draw(False,txt)
            K_plus.search(n_limit)
        return 
        
    # a version of simple-check that displays the necessary steps and computations along the way
    # it also provided plots for comparing the different bounds
    def detailed_check(self,diagonal_only=False):
        print("kernel:")
        self.draw()
        print("Gamma_target:",self.Ga_target)
        print("adjacency matrix:")
        print(self.A)
        print()
        print('P(\u03BB)=',self.P,'\n')
        print('P(\u03BB)B(\u03BB):')
        for row in self.adj:
              print(row)
        print()
        print("active vertices:",self.active)
        print("possible subsets:",self.Uc)
        print("la_H for kernel: {:.4f}".format(self.la_H))
        print()

        la_max=self.la_H+0.2

        self.gas=[self.la_U(self.Uc[i],True) for i in range(self.nr)]

        for i in range(self.nr):
            print("U: {}    \u03BB_U: {:.4f}".format(self.Uc[i],self.las[i]))
            print('P B~_o,U:',self.Bt[i][self.root])
            print('P s_U:',self.s[i])
            print('----------------------------------------------------')

        pt_base=point([(self.la_H,self.Ga_H)], color='black')

        for i1 in range(self.nr):
            i2_bnd=i1+1 if diagonal_only else self.nr
            for i2 in range(i1,i2_bnd):
                lat=max(self.las[i1],self.las[i2])
                cUV=sum(self.Bt[i1][v]*self.Bt[i2][v] for v in range(self.n))
                num=(self.s[i1]+self.P)*(self.s[i2]+self.P)
                denum=cUV+(self.Bt[i1][self.root]+self.Bt[i2][self.root])*self.P/2
                denum_or=cUV+self.P^2 if i1==i2 else cUV
                
                print('U={}   V={}   \u03BB~: {:.4f}'.format(self.Uc[i1],self.Uc[i2],lat))
                print('P^2 c_U,V:',cUV,'\n')

                Q=num-self.Ga_target*denum
                print('Q(\u03BB)=',Q,'\n')
                Qt=Q(ka_var+lat)
                print('Q~(\u03BA)=Q(\u03BB+\u03BB~)=',Qt,'\n')
                print("simple-check", "SUCCESSFUL" if check_coeffs(Qt) else "FAILED")

                fig=plot([num(x)/denum(x),self.Ga_target],self.la_H,la_max)
                fig+=plot(num(x)/denum_or(x),self.la_H,la_max, color='red')
                fig+=line([(lat,num(lat)/denum_or(lat)),(lat,self.Ga_target),(lat,num(lat)/denum(lat))],color='black')
                fig+=pt_base
                if i1==i2:
                    fig+=point([(self.las[i1],self.gas[i1])], color='black')
                fig.show()

    #def x_H(self,v,la):
    #    return self.adj[v](la)/self.P(la)
        
    def tail_functions(self,v,print_mode=False):
        la_infty,Ga_infty=self.ev_infty(v)
        r_infty=r(la_infty)
        S=sum(self.adj[v])/self.P
        T=sum(entry^2 for entry in self.adj[v])/self.P^2
        Sh=S(t+1/t)
        Th=T(t+1/t)
        Jh=(Sh+t/(t-1))^2/(Th+t^2/(t^2-1))
        if print_mode:
            print("v={}".format(v))
            print('P(\u03BB)=',self.P,'=',self.P.factor())
            print()
            print('P(\u03BB)*B_u,v(\u03BB):')
            for u in range(self.n):
                print('u={}: '.format(u),self.adj[u,v],'=',self.adj[u,v].factor())
            print()
            print('S(\u03BB)=',S)
            print('S-hat(t)=S(t+1/t)=',Sh)
            print()
            print('T(\u03BB)=',T)
            print('T-hat(t)=T(t+1/t)=',Th)
            print()
            print('J-hat(t)=',Jh)        
            print()
            
        return la_infty,Ga_infty,r_infty,S,T,Sh,Th,Jh

    # checks the conditions of Lemma 7.5
    def tail_check_lower(self,v,k0=1,show_plots=True):
        la_infty,Ga_infty,r_infty,S,T,Sh,Th,Jh=self.tail_functions(v,True)
        
        # instead, we may check that 
        # 1) the derivative of J at la_infty is positive
        # 2) f(la_infty)>0
        # then look for k for which these hold for la_0 as well
        
        H0=self.H.copy()
        add_path(H0,k0,v)
        la0=lambda_G(H0)
        r0=r(la0)

        #print("the relevant interval for \u03BB:          I=({:.4f},{:.4f})".format(la0,la_infty))
        #print("the relevant interval for t=r(\u03BB):  r(I)=({:.4f},{:.4f})".format(r0,r_infty))
        #print()
        
        # cond (i)
        cond1=is_monotone(Jh,r0,r_infty)
        print('condition (i): J(\u03BB)=J-hat(r(\u03BB)) should be monotone increasing on ({:.4f},{:.4f}):'.format(la0,la_infty),cond1)            
        if show_plots:
            plot(Jh(r(x)),la0,la_infty).show()
                        
        # cond (ii)
        fh=(Th+1)*t-(Sh+t/(t-1))
        #print('f-hat(t)=',fh)
        cond2=is_positive(fh,r0,r_infty)
        print('condition (ii): f(\u03BB)=f-hat(r(\u03BB)) should be positive on ({:.4f},{:.4f}):'.format(la0,la_infty),cond2)
        if show_plots:
            plot(fh(r(x)),la0,la_infty).show()
            
    # checks the conditions of Theorem 8.1
    def tail_check_upper(self,v=None,par=(0,0,0),k_limit=10,show_plots=True,eps=1e-8):
        if v==None:
            v=self.n-1
        la_infty,Ga_infty,r_infty,S,T,Sh,Th,Jh=self.tail_functions(v,True)
        
        lap,lapp,c=par
        if lap==0:
            # old lapp:
            # old_lapp=eps+find_root(self.adj[self.root,v]-self.adj[v,v],la_infty,la_infty+4)
            lapp=eps+find_root(self.adj[self.root,v]-self.P,la_infty,la_infty+4)
            lap=eps+find_root((S+1)^2/(T+1)-Ga_infty,la_infty,lapp)
            rp=r(lap)
            fun=lambda x: (S(la_infty)+1+x)^2/(T(la_infty)+1+x^2)-Ga_infty
            c=1 if fun(1)>=0 else eps+find_root(fun,1,16)
            print("setting  \u03BB'={:.5f}  \u03BB''={:.5f}  c={:.5f}".format(lap,lapp,c))  
        rp=r(lap)

        # cond (i)
        val=2*la_infty+3
        cond1=val>Ga_infty
        print("condition (i): {:.4f} > {:.4f}  --> {}".format(val,Ga_infty,cond1))

        # old cond (iv), now new version is cond (ii)
        #fun4=self.adj[v,v]-self.adj[self.root,v]
        #cond4=fun4(lapp)>0 and is_positive_pol(fun4,lapp,np.inf)
        #print("condition (iv):",cond4)
        #if show_plots:
        #    plot(fun4(x)/self.P(x),lapp,lapp+5).show()
        
        # cond (ii)
        val=self.adj[self.root,v](lapp)/self.P(lapp)
        cond2=val<=1
        print("condition (ii): {:.4f} <= 1  --> {}".format(val,cond2))

        # cond (iii)
        val=self.adj[v,v](lap)/self.P(lap)
        cond3=val>1/r_infty
        print("condition (iii): {:.4f} > {:.4f} --> {}".format(val,1/r_infty,cond3))

        # cond (iv)
        fun4=(S+1)^2/(T+1)
        cond4=fun4(lap)>Ga_infty and fun4(lapp)>Ga_infty and is_positive(fun4-Ga_infty,lap,lapp)
        print("condition (iv):",cond4)
        if show_plots:
            plot([fun4(x),Ga_infty],lap,lapp).show()

        # cond (v)
        cond5=is_monotone(Jh,r_infty,rp)
        print("condition (v): J(\u03BB)=J-hat(r(\u03BB)) should be monotone increasing on ({:.4f},{:.4f}): {}".format(la_infty,lap,cond5))
        if show_plots:
            plot([Jh(r(x)),Ga_infty],la_infty,lap).show()

        # cond (vi) and (vii)
        fun6=(Sh+1+c)^2/(Th+1+c^2)        
        cond6=is_positive(fun6-Ga_infty,r_infty,rp)
        fun7=(Sh+1+c*t)^2/(Th+1+(c*t)^2)
        cond7=is_positive(fun7-Ga_infty,r_infty,rp)
        print("condition (vi):",cond6)
        print("condition (vii):",cond7)        
        if show_plots:
            plot([fun6(x),fun7(x),Ga_infty],r_infty,rp).show()
        
        # cond (viii)
        for k in range(2,k_limit+1):
            fun8=(2*(Sh+t/(t-1))*(1-(t+1)/(t-1)*t^(-k))/Ga_infty-2*k*t^(-k))/(t^3/(t^2-1))
            if fun8(r_infty)>c and fun8(rp)>c:
                cond8=is_positive(fun8-c,r_infty,rp)
                print("condition (viii):",cond8,"for k={}".format(k))
                if show_plots:
                    plot([fun8(x),c],r_infty,rp).show()
                return
        print("condition (viii) failed up to k=",k_limit)        
        
        
# for showing that the tail must be a path
# a path is added to H, then a cherry/triangle is added at the end of the path
# then it is verified that all extensions have Gamma above the limiting ratio 
def tail_check_upper_small(H,path_len,root=0,v=0,triangle=False,tree_mode=False):
    H1=H.copy()
    add_path(H1,path_len,v)
    m=H1.order()
    H1.add_edge(m-1,m)
    H1.add_edge(m-1,m+1)
    if triangle:
        H1.add_edge(m,m+1)        
    K=Kernel(H1,root,0,tree_mode)
    K.draw()

    print(K.Ga_H,">", K.Ga_target, K.Ga_H>K.Ga_target)

    K.active=[m-1,m,m+1]
    K.act_Uc()
    print("extensions > {}: {}".format(K.Ga_target,K.simple_check()))
    

print("limiting ratios:")
print(Ga_conn,"(connected graphs)")
print(Ga_tree,"(trees)")
          
#for H in [graphs.CompleteGraph(4),graphs.StarGraph(4),graphs.StarGraph(5),graphs.StarGraph(3)]:
#    K=Kernel(H)
#    print(K.ev_infty(0))

