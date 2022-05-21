__all__ = ['matrix', 'vector']
from .matrix import Matrix

def eqn_solve(a,b):
    """Solves system of linear equations Ax=B using Gauss-Jordan Elimination"""
    m=a.m
    a=a.concat(b) # augment the matrix
    inconsistent=False
    r=0
    for c in range(m):
        if r>=a.n: break
        j=r
        while j<a.n and a[j][c]==0:
            j+=1
        if j>=a.n:
            # return 'Infinite or no solution'
            inconsistent=True
            continue
        a[r],a[j]=a[j],a[r]
        x=a[r][c]
        for c2 in range(a.m):
            a[r][c2]/=x
        for r2 in range(a.n):
            if r2!=r:
                x=a[r2][c]
                for c2 in range(a.m):
                    a[r2][c2]-=x*a[r][c2]
        r+=1
    if inconsistent:
        return f'Infinite Solutions or no solution. RREF:\n{augmented_toStr(a,m)}'
    return a[:,m:]


def bestfit_line(a:Matrix,b:Matrix):
    """A^{T}Ax=A^{T}b"""
    x=eqn_solve(a.T*a,a.T*b)
    return x


def augmented_toStr(a:Matrix,m:int):
    s=[]
    col_len=[max(len(f'{a[j][i]+0:n}') for j in range(a.n)) + 1 for i in range(a.m)]
    for i in range(a.n):
        l=[]
        for j in range(len(a[0])):
            z=f'{a[i][j]+0:n}'
            x=col_len[j]-len(z)
            l.append(' '*(x//2)+z+' '*(x-x//2))
            if j+1==m:
                l.append('| ')
        s.append(' ['+''.join(l)[:-1]+']')
    s[0]='['+s[0][2:]
    return '['+'\n'.join(s)+']'
