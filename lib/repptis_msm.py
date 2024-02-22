import numpy as np

def construct_M(p_mm, p_mp, p_pm, p_pp, NS, N):
    """Construct transition matrix M"""
    # N -- number of interfaces
    # NS -- dimension of MSM, 4*N-5 when N>=4
    # p_mp -- list shape (N-1,) with local crossing probability minus-to-plus (mp)
    
    assert N>=4
    assert N==len(p_mm)+1
    assert N==len(p_mp)+1
    assert N==len(p_pm)+1
    assert N==len(p_pp)+1
    assert NS==4*N-5

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


def global_cross_prob(M, doprint=False):
    # probability to arrive in -1 before 0
    # given that you are at 0 now and that you are leaving 0
    # = crossing probability from 0 to -1

    NS = len(M)
    assert NS>2

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
    y1 = np.dot(M11,z1) + np.dot(E,z2)
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


def vector_G(M, tau1, taum, tau2, doprint=False):
    taum2 = taum + tau2    # TODO
    NS = len(M)
    assert NS>2
    
    # take pieces of transition matrix
    Mp = M[1:-1,1:-1]
    a = np.identity(NS-2)-Mp   # 1-Mp
    a1 = np.linalg.inv(a)      # compute (1-Mp)^(-1)

    # other pieces
    D   = M[1:-1, np.array([0,-1])]
    E   = M[np.array([0,-1]), 1:-1]
    M11 = M[np.array([0,-1]), np.array([0,-1])]

    # part tau
    tau_1 = np.array([taum2[0], taum2[-1]]).reshape(2,-1)
    tau_p = taum2[1:-1].reshape(len(taum2)-2,-1)

    # compute G vector
    g1 = np.array([[0],[0]])    # + tau_1            # not filled in correctly!! so zero anyways
    g2 = np.dot(a1, np.dot(D,g1) + tau_p)

    # compute H vector
    h1 = np.dot(M11,g1) + np.dot(E,g2) + tau_1
    h2 = np.dot(D,g1) + np.dot(Mp,g2)  + tau_p

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
        print("vector tau m2")  # TODO
        print(tau_1)
        print(tau_p)
        print("vector z1,z2")
        print(g1)
        print(g2)
        print("vector y1,y2")
        print(h1)
        print(h2)
        print("check", np.sum((g2-h2)**2))  # 0, so g2 and h2 indeed the same
    return g1, g2, h1, h2


