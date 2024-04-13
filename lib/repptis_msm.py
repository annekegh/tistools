import numpy as np

def construct_M(p_mm, p_mp, p_pm, p_pp, N):
    """Construct transition matrix M"""
    # N -- number of interfaces
    # NS -- dimension of MSM, 4*N-5 when N>=3
    # p_mp -- list shape (N-1,) with local crossing probability minus-to-plus (mp)
    
    assert N>=3
    assert N==len(p_mm)+1
    assert N==len(p_mp)+1
    assert N==len(p_pm)+1
    assert N==len(p_pp)+1
    NS = 4*N-5

    if N==3:
        return construct_M_N3(p_mm, p_mp, p_pm, p_pp, NS)

    # construct transition matrix
    M = np.zeros((NS,NS))
    
    # states [0-] and [0+-]
    M[0,1:4] = [p_mm[0],p_mp[0],0]
    M[1,0] = 1
    M[2,4:8] = [p_mm[1],p_mp[1],0,0]
    M[3,0] = 1

    # states [1+-] special
    M[4,3] = 1
    M[6,3] = 1

    # states [(N-2)+-] special
    M[(N-2)*4,-5:-3] = [p_pm[N-3],p_pp[N-3]]
    M[(N-2)*4+1,-1] = 1

    # state B=N-1 special
    M[-1,0] = 1

    for i in range(1,N-2):
        #print("starting from state i",i)
        M[1+4*i,4*(i+1):4*(i+1)+2] = [p_mm[i+1],p_mp[i+1]]
        M[3+4*i,4*(i+1):4*(i+1)+2] = [p_mm[i+1],p_mp[i+1]]

    for i in range(2,N-2):
        #print("starting from state i",i)
        M[4*i,  4*i-2:4*i] = [p_pm[i-1],p_pp[i-1]]
        M[4*i+2,4*i-2:4*i] = [p_pm[i-1],p_pp[i-1]]

    return M

def construct_M_N3(p_mm, p_mp, p_pm, p_pp, NS):
    """Construct transition matrix M for N=3"""

    # construct transition matrix
    M = np.zeros((NS,NS))

    # states [0-] and [0+-]
    M[0,1:4] = [p_mm[0],p_mp[0],0]
    M[1,0] = 1
    M[2,4:6] = [p_mm[1],p_mp[1]]   # changed
    M[3,0] = 1

    # states [1+-] special
    M[4,3] = 1
    # states [(N-2)+-] special
    M[5,-1] = 1
    # state B=N-1 special
    M[-1,0] = 1

    return M


def construct_M_milestoning_dir(p_mm, p_mp, p_pm, p_pp, N):
    #TODO
    """Construct transition matrix M"""
    # N -- number of interfaces
    # NS -- dimension of MSM, N when N>=3
    # p_mp -- list shape (N-1,) with local crossing probability minus-to-plus (mp)
    
    assert N>=4 # maybe N=3??? TODO
    assert N==len(p_mm)+1
    assert N==len(p_mp)+1
    assert N==len(p_pm)+1
    assert N==len(p_pp)+1
    
    # NS: 0 to 0, 0 to 1, 1 to 2, ..., N-2 to N-1 (=B)
    # and in two directions
    NS = 2*N

    #if N==3:
    #    return construct_M_N3(p_mm, p_mp, p_pm, p_pp, NS)

    # construct transition matrix
    M = np.zeros((NS,NS))
    
    # states = lambda0-, lambda0+, lambda1-, lambda+, ...
    # lambda0
    M[0,1] = 1
    M[1,0] = p_mm[0]
    M[1,2] = p_mp[0]
    
    # lambda1
    # lambda i- becomes lambda (i-1)- or lambda (i+1)+
    # index 2*i        index 2*(i-1)     index 2*(i+1)+1
    # lambda i+ becomes lambda (i-1)- or lambda (i+1)+
    for i in range(1,N-1):
        M[2*i,2*(i-1)]   = p_mm[1]
        M[2*i,2*(i+1)+1] = p_mp[1]
        M[3,3] = p_pm[1]
        M[3,3] = p_pp[1]
    
    # for i=N-2
    [p_mm[0],p_mp[0],0]
    M[1,0] = 1
    M[2,4:8] = [p_mm[1],p_mp[1],0,0]
    M[3,0] = 1

    # states [1+-] special
    M[4,3] = 1
    M[6,3] = 1

    # states [(N-2)+-] special
    M[(N-2)*4,-5:-3] = [p_pm[N-3],p_pp[N-3]]
    M[(N-2)*4+1,-1] = 1

    # state B=N-1 special
    M[-1,0] = 1

    for i in range(1,N-2):
        #print("starting from state i",i)
        M[1+4*i,4*(i+1):4*(i+1)+2] = [p_mm[i+1],p_mp[i+1]]
        M[3+4*i,4*(i+1):4*(i+1)+2] = [p_mm[i+1],p_mp[i+1]]

    for i in range(2,N-2):
        #print("starting from state i",i)
        M[4*i,  4*i-2:4*i] = [p_pm[i-1],p_pp[i-1]]
        M[4*i+2,4*i-2:4*i] = [p_pm[i-1],p_pp[i-1]]

    return M




def global_cross_prob(M, doprint=False):
    # probability to arrive in -1 before 0
    # given that you are at 0 now and that you are leaving 0
    # = crossing probability from 0 to -1
    # The interesting answer is y1[0] = Pcross-global

    NS = len(M)
    assert NS>=3

    # take pieces of transition matrix
    Mp = M[1:-1,1:-1]
    a = np.identity(NS-2)-Mp    # 1-Mp
    a1 = np.linalg.inv(a)       # (1-Mp)^(-1)

    # other pieces
    D = M[1:-1, np.array([0,-1])]
    E = M[np.array([0,-1]), 1:-1]
    M11 = M[np.array([0,-1]),np.array([0,-1])]

    # compute Z vector
    z1 = np.array([[0],[1]])
    z2 = np.dot(a1,np.dot(D,z1))

    # compute H vector
    y1 = np.dot(M11,z1) + np.dot(E,z2)  # y1[0] = Prcross
    y2 = np.dot(D,z1) + np.dot(Mp,z2)

    if doprint:
        print("Mp eigenvals")
        vals, vecs = np.linalg.eig(Mp)
        print(vals)
        print("1-Mp eigenvals")
        vals, vecs = np.linalg.eig(a)
        print(vals)
        #print(np.dot(a,a1)  # identity matrix indeed
        print("other pieces M")
        print(D)
        print(E)
        print(M11)
        print("vector z1,z2")
        print(z1)
        print(z2)
        print("vector y1,y2")
        print(y1)
        print(y2)
        print("check", np.sum((y2-z2)**2))  # 0, so z2 and y2 indeed the same
    return z1, z2, y1, y2



def vector_GH(M, tau1, taum, tau2, absor, kept, doprint=False):
    taum2 = taum + tau2
    NS = len(M)
    assert NS>=3
    
    assert len(M) == len(absor)+len(kept)

    # take pieces of transition matrix    
    Mp  = np.take(np.take(M, kept, axis=0), kept, axis=1)
    # other pieces
    D   = np.take(np.take(M, kept, axis=0), absor, axis=1)
    E   = np.take(np.take(M, absor, axis=0), kept, axis=1)
    M11 = np.take(np.take(M, absor, axis=0), absor, axis=1)
    #print(D.shape, E.shape, M11.shape, Mp.shape, absor.shape, kept.shape)

    a = np.identity(len(Mp))-Mp   # 1-Mp
    a1 = np.linalg.inv(a)         # compute (1-Mp)^(-1)

    # part tau (m2)
    t1 = taum2[absor].reshape(len(absor),1)
    tp = taum2[kept].reshape(len(kept),1)
    # part tau (m). the middle part
    st1 = taum[absor].reshape(len(absor),1)
    stp = taum[kept].reshape(len(kept),1)

    # compute G vector: DESIGN!
    g1 = np.zeros((len(absor),1))  # + t1 # t1 is set to zero
    g2 = np.dot(a1, np.dot(D,g1) + tp)

    # compute H vector
    h1 = np.dot(M11,g1) + np.dot(E,g2) + t1
    h2 = np.dot(D,g1) + np.dot(Mp,g2)  + tp

    # EXTRA: remove middle part m of first time
    h1 -= st1
    h2 -= stp

    if doprint:
        print("Mp eigenvals")
        vals, vecs = np.linalg.eig(Mp)
        print(vals)
        print("1-Mp eigenvals")
        vals, vecs = np.linalg.eig(a)
        print(vals)
        #print(np.dot(a,a1)  # identity matrix indeed
        print("other pieces M")
        print(D)
        print(E)
        print(M11)
        print("vector tau m2")
        print(t1)
        print(tp)
        print("vector g1,g2")
        print(g1)
        print(g2)
        print("vector h1,h2")
        print(h1)
        print(h2)
        print("check", np.sum((g2-h2)**2))  # 0, so g2 and h2 indeed the same
    return g1, g2, h1, h2


def vector_G(M, tau1, taum, tau2, doprint=False):
    taum2 = taum + tau2
    NS = len(M)
    assert NS>2

    absor = np.array([0,NS-1])
    kept  = np.array([i for i in range(NS) if i not in absor])

    g1, g2, h1, h2 = vector_GH(M, tau1, taum, tau2, absor, kept, doprint=doprint)
    return g1, g2, h1, h2


#======================================
# Trying other MFPTs
#======================================

# NOT FINISHED
def mfpt_one_target(M, tau1, taum, tau2, target, doprint=False):
    NS = len(M)
    assert NS>=3
    assert isinstance(target,int)

    absor = np.array([target])
    #print("special absor", absor)
    kept  = np.array([i for i in range(NS) if i not in absor])
    #print("special kept", kept)

    g1, g2, h1, h2 = vector_GH(M, tau1, taum, tau2, absor, kept, doprint=doprint)
    return g1, g2, h1, h2


def vector_Gbis2(M, tau1, taum, tau2, i, j, doprint=False):
    NS = len(M)
    assert NS>=3
    assert isinstance(i,int)
    assert isinstance(j,int)

    absor = [i,j]
    if True:
        absor.append(0)
        absor.append(NS-1)
    absor = list(set(absor))
    absor = sorted(absor)
    kept  = np.array([i for i in range(NS) if i not in absor])
    g1, g2, h1, h2 = vector_GH(M, tau1, taum, tau2, absor, kept, doprint=doprint)
    return g1, g2, h1, h2, absor, kept


def vector_Gbis(M, tau1, taum, tau2, i, j, doprint=False):
    taum2 = taum + tau2
    NS = len(M)
    assert NS>=3
    N = int((NS+5)/4)
    
    #assert i<j   # TODO

    indices = []
    bounds = []
    cut = []
    # convert i lambda_i [i+-] to indices in M
    
    # maybe put i==0 also into i=1..N-3 range!!!!
    if i==0:   # cross lambda_0
        bounds.append(0)
        first = 1        #=>practically
    elif i==N-2:
        bounds.append(4*i)    # (N-2)+- LML
        bounds.append(4*i+1)  # (N-2)+- LMR
        first = NS-1  # weird!!!
        first = 90000
        cut.extend(range(4*i))
    elif i==N-1:
        bounds.append(NS-1)
        first = 10000000000
    else:   # i=1..N-3
        bounds.append(4*i)    # i+- LML
        bounds.append(4*i+1)  # i+- LMR
        bounds.append(4*i+2)  # i+- RML
        bounds.append(4*i+3)  # i+- RMR
        first = 4*i+4
        cut.extend(range(4*i))
    #elif i>=N-2:
    #    raise ValueError("value too high")
    #elif i==1:  # cross lambda_1
    #    bounds.append(2)
    #    first = 3

    if j==0:   # cross lambda_0
        bounds.append(0)
        last = 1        #=>practically
    elif j==N-2:
        bounds.append(4*j)    # (N-2)+- LML
        bounds.append(4*j+1)  # (N-2)+- LMR
        last = NS-4
        cut.extend(range(4*j+1,NS))
    elif j==N-1:
        bounds.append(NS-1)
        last = NS-2
    else:    #j=1...N-3
        bounds.append(4*j)    # j+- LML
        bounds.append(4*j+1)  # j+- LMR
        bounds.append(4*j+2)  # j+- RML
        bounds.append(4*j+3)  # j+- RMR
        last = 4*j-1

    bounds.append(0)
    bounds.append(NS-1)
    #first = min(bounds)
    #last = max(bounds)
    
    absor = list(set(bounds))
    absor = sorted(absor)
    
    #if NS-1 not in absor:
        
    
    #if first==last:
    kept  = np.array([i for i in range(NS) if i not in absor])
    #else:
    #kept  = np.array([i for i in range(first,last+1) if i not in absor])
    print(kept)
    print(absor)
    
    #absor = np.array(indices)
    #kept = np.array(select)
    g1, g2, h1, h2 = vector_GH(M, tau1, taum, tau2, absor, kept, doprint=doprint)
    return g1, g2, h1, h2, absor, kept

if False: 
    # take pieces of transition matrix
    Mp = np.take(np.take(M, select, axis=0), select, axis=1)
    print(Mp)
    a = np.identity(NS-len(indices))-Mp   # 1-Mp
    a1 = np.linalg.inv(a)      # compute (1-Mp)^(-1)

    # other pieces
    D   = np.take(np.take(M, select, axis=0), indices, axis=1) #as if M[select, indices]
    E   = np.take(np.take(M, indices, axis=0), select, axis=1)
    M11 = np.take(np.take(M, indices, axis=0), indices, axis=1)

    # part tau
    tau_1 = np.take(taum2, indices, axis=0).reshape(len(indices),-1)
    tau_p = np.take(taum2, select, axis=0).reshape(len(select),-1)

    # compute G vector
    g1 = np.array([[0],[0]])    # + tau_1            # not filled in correctly!! so zero anyways
    g1 = np.zeros((len(indices),1))
    g2 = np.dot(a1, np.dot(D,g1) + tau_p)



    
def get_pieces_matrix(M, absor, kept):
    NS = len(M)
    assert len(set(absor))==len(absor)
    assert min(absor)>=0
    assert max(absor)<NS

    # TODO check kept

    # take pieces of transition matrix
    Mp  = np.take(np.take(M, kept, axis=0), kept, axis=1)
    # other pieces
    D   = np.take(np.take(M, kept, axis=0), absor, axis=1)
    E   = np.take(np.take(M, absor, axis=0), kept, axis=1)
    M11 = np.take(np.take(M, absor, axis=0), absor, axis=1)
    #print(D.shape, E.shape, M11.shape, Mp.shape, absor.shape, kept.shape)

    # TODO
    if len(absor)==1:
        print("D.shape, E.shape, M11.shape, Mp.shape, absor.shape, kept.shape")
        print(D.shape, E.shape, M11.shape, Mp.shape, absor.shape, kept.shape)
        raise Error #reshape all

    return Mp, D, E, M11

def get_pieces_vector(vec, absor, kept):
    NS = len(vec)
    # check no double indices
    assert len(set(absor))==len(absor)
    assert len(set(kept))==len(kept)
    # check no indices in both sets TOOD
    
    assert min(absor)>=0
    assert max(absor)<NS

    assert len(kept)+len(absor)==NS

    # take pieces of vector
    v1 = vec[absor].reshape(len(absor),1)
    v2 = vec[kept].reshape(len(kept),1)
    return v1, v2

def create_labels_states(N):
    assert N>=3
    labels1 = ["0-     ","B      "]
    labels2 = ["0+- LML","0+- LMR","0+- RML","1+- LML","1+- LMR"]
    if N>3:
        for i in range(1,N-2):
            labels2.append(str(i)  +"+- RML")
            labels2.append(str(i)  +"+- RMR")
            labels2.append(str(i+1)+"+- LML")
            labels2.append(str(i+1)+"+- LMR")
    return labels1, labels2

def create_labels_states_all(N):
    assert N>=3
    labels = ["0-     ","0+- LML","0+- LMR","0+- RML",
        "1+- LML","1+- LMR"]
    if N>3:
        for i in range(1,N-2):
            labels.append(str(i)  +"+- RML")
            labels.append(str(i)  +"+- RMR")
            labels.append(str(i+1)+"+- LML")
            labels.append(str(i+1)+"+- LMR")
    labels.append("B      ")
    return labels


def print_vector(g, states=None, sel=None):
    #print(g)
    #print(states)
    #print(sel)
    for i in range(len(g)):
        if states is None:
            print("state", i, g[i])
        else:
            if sel is None:
                print("state", states[i], g[i][0])
            else:
                print("state", states[sel[i]], g[i][0])
