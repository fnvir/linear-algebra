import numbers

__all__=['DimensionError','Matrix']


class DimensionError(ValueError):
    """Inaccurate dimension of matrix for specific operations"""
    def __init__(self,err='Dimensions do not match for this operation'):
        super().__init__(err)


class Matrix:
    """
    Matrix
    """
    def __init__(self,_matrix,fractional=False):
        """
        :param _matrix: 1D or 2D iterables containing numbers or str representing numbers
        :param fractional: bool whether to map every item to Fractions or not

        Examples
        --------
        >>> Matrix([1,2,3])
        [[1]
         [2]
         [3]]

        >>> Matrix([['22/7','3.14',26],['1/4','-1/2',6.5]],True)
        [[22/7 157/50  26 ]
         [ 1/4  -1/2  13/2]]

        >>> Matrix([2]*5 for i in range(3))
        [[2 2 2 2 2]
         [2 2 2 2 2]
         [2 2 2 2 2]]
        """
        self.__a=[]
        for i in _matrix:
            if hasattr(i,'__getitem__') and type(i)!=str: self.__a.append([j for j in i])
            else: self.__a.append([i])
            fractional|=any(type(j)==str for j in self.__a[-1])
        self.n,self.m= len(self.__a), len(self.__a[0])
        self.dim=(self.n,self.m)
        if fractional:
            from fractions import Fraction
            self.__a=[[Fraction(j) for j in i] for i in self]

    @classmethod
    def from_order(cls,r:int,c:int,_fill=0):
        """Constructs a matrix of dimension (r,c) filled with '_fill'"""
        return cls([_fill]*c for i in range(r))

    @staticmethod
    def identity(n:int):
        """Constructs an identity matrix of size n"""
        return Matrix([int(i==j) for j in range(n)] for i in range(n))

    def transpose(self):
        """as the name says"""
        return Matrix([self[j][i] for j in range(self.n)] for i in range(self.m))

    def determinant(self):
        """
        Computes determinant by converting into upper-triangular matrix
        and returning the product of the elements in the principal diagonal\n
        Complexity: O(n^3)
        """
        if self.n!=self.m:
            raise DimensionError('Not a square matrix')
        A=self.__copy__()
        _det=1
        for c in range(self.n):
            r=c
            while r<self.n and A[r][c]==0:
                r+=1
            if r>=self.n:
                return 0
            A[c],A[r] = A[r],A[c]
            if (r-c)%2:
                _det*=-1
            for r2 in range(c+1,self.n):
                x=A[r2][c]
                for c2 in range(self.n):
                    A[r2][c2]-=(x*A[c][c2]/A[c][c])
            _det*=A[c][c]
        return _det

    def trace(self):
        if self.n!=self.m:
            return None
        return sum(self[i][i] for i in range(self.n))

    @property
    def T(self):
        """The transpose of the matrix"""
        return self.transpose()

    @property
    def det(self):
        """Determinant"""
        try:
            return self.determinant()
        except ValueError:
            return None

    def multiply(a,b):
        n1,m1=a.dim
        n2,m2=b.dim
        c=Matrix.from_order(n1,m2)
        if m1!=n2: raise DimensionError(f'Multiplication undefined for dimensions {a.dim} and {b.dim}')
        for i in range(m2):
            for j in range(n1):
                for k in range(n2):
                    c[j][i]+=b[k][i]*a[j][k]
        return c

    def inverse(self):
        """Inverse of a matrix using Gauss-Jordan"""
        if self.det==0:
            raise ValueError('Singular Matrix')
        x=self | self.identity(self.n)
        x.m=self.m
        return x.rref()[:, self.m:]

    def __echelon_form(self,reduced=False):
        """
        Calculates:
            REF using Gaussian Elimination\n
            RREF using Gauss-Jordan Elimination
        Complexity: O(n^3)
        """
        A=self.__copy__()
        r=0
        # self.m used in case of augmented matrix
        for c in range(self.m):
            if r>=A.n:
                break
            j=r
            while j<A.n and A[j][c]==0:
                j+=1
            if j>=A.n:
                continue
            A[r],A[j] = A[j],A[r]
            for r2 in range(r+1,A.n):
                x=A[r2][c]
                for c2 in range(A.m):
                    A[r2][c2]-=x*A[r][c2]/A[r][c]
            if reduced:
                for r2 in range(r):
                    x=A[r2][c]
                    for c2 in range(A.m):
                        A[r2][c2]-=x*A[r][c2]/A[r][c]
                x=A[r][c]
                for c2 in range(A.m):
                    A[r][c2]/=x
            r+=1
        # print('Rank:',r)
        return A

    def ref(self):
        return self.__echelon_form()

    def rref(self):
        return self.__echelon_form(True)

    def lower_upper(self):
        """LU Decomposition using Doolittle algorithm\n
        Complexity: O(n^3)"""
        if not self.is_square():
            raise NotImplementedError('Not implemented for non-square matrix')
        L=Matrix.from_order(*self.dim)
        U=Matrix.from_order(*self.dim)
        for i in range(self.n):
            for j in range(i,self.n):
                s=sum(L[i][k]*U[k][j] for k in range(i))
                U[i][j]=self[i][j]-s
            for j in range(i,self.n):
                if i==j:
                    L[i][j]=1
                else:
                    s=sum(L[j][k]*U[k][i] for k in range(i))
                    if U[i][i]==0: raise ValueError('LU Decomposition not possible')
                    L[j][i]=(self[j][i]-s)//U[i][i]
        return L,U

    @staticmethod
    def projection(__a,__b):
        """Matrix projection = soln of A^{T}Ax=A^{T}b"""
        p=(__a.T*__a) | (__a.T*__b)
        p.m=__a.m
        return p.rref().col_slice(p.m,None)

    def concat(self, other):
        """ Concatenates two matrices containing the same number of rows"""
        if not isinstance(other,Matrix):
            raise TypeError('Use commonsense')
        if self.n!=other.n:
            raise DimensionError
        return Matrix(self[i]+other[i] for i in range(self.n))

    def map(self,f):
        return Matrix([f(j) for j in i] for i in self)

    def is_ref(self):
        """Checks if the matrix is in row echelon form"""
        return self.__is_echelon_form()

    def is_rref(self):
        """Checks if the matrix is in reduced row echelon form"""
        return self.__is_echelon_form(True)

    def __is_echelon_form(self,reduced=False):
        c=0
        all_zeroes=False
        for r in range(self.n):
            j=c
            while j<self.m and self[r][j]==0:
                j+=1
            if j>=self.m:
                all_zeroes=True
                continue
            if all_zeroes:
                return False
            c=j
            for r2 in range(r+1,self.n):
                if self[r2][c]!=0:
                    return False
            if reduced:
                if self[r][c]!=1:
                    return False
                for r2 in range(r):
                    if self[r2][c]!=0:
                        return False
            c+=1
        return True

    def is_singular(self):
        """Checks if the matrix is singular(Square matrix with determinant=0)"""
        return self.is_square() and self.det==0

    def is_square(self):
        return self.n==self.m

    def is_diagonal(self):
        if self.n!=self.m:
            return False
        for i in range(self.n):
            for j in range(self.n):
                if i!=j and self[i][j]!=0:
                    return False
        return True

    def is_upper_triangular(self):
        return self.is_square() and all(self[i][j]==0 for i in range(self.n) for j in range(i))

    def is_lower_triangular(self):
        return self.is_square() and all(self[i][j]==0 for i in range(self.n) for j in range(i+1,self.m))

    def row_slice(self,*args):
        return Matrix(self.__a[slice(*args)])

    def col_slice(self,*args):
        return Matrix(i[slice(*args)] for i in self)

    def __slice(self, r, c):
        x,y=type(r),type(c)
        if x==y==int:
            return self.__a[r][c]
        if x==int: r=slice(r,r+1)
        if y==int: c=slice(c,c+1)
        return Matrix(i[c] for i in self.__a[r])

    def __len__(self):
        return self.n

    def __setitem__(self,i,x):
        self.__a[i]=list(x) # .copy()

    def __getitem__(self,k):
        t=type(k)
        if t==int:
            return self.__a[k]
        if t==slice:
            return self.row_slice(k.start,k.stop,k.step)
        if t!=tuple or len(k)>2:
            raise TypeError('Matrix indices must be ints or slices or tuples(of max size 2)')
        if len(k)==1:
            return self.__slice(k[0],slice(None))
        return self.__slice(*k)

    def __neg__(self):
        return self*-1

    def __pos__(self):
        return self

    def __reversed__(self):
        return self[::-1,::-1]

    def __add__(self,other):
        if not isinstance(other,Matrix):
            raise TypeError(f"Operation undefined for {type(self)} and {type(other)}")
        if self.n!=other.n or self.m!=other.m: raise Exception('Dimension Error')
        return Matrix([self[i][j]+other[i][j] for j in range(self.m)] for i in range(self.n))

    def __radd__(self, other):
        return self+other

    def __sub__(self,other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return self+(-other)

    def __mul__(self,other):
        if isinstance(other,(int,float)):
            return Matrix([j*other for j in i] for i in self)
        if isinstance(other,Matrix):
            return self@other
        raise TypeError('Unsupported Operand')

    def __rmul__(self, other):
        return self*other

    def __matmul__(self, other):
        return self.multiply(other)

    def __truediv__(self, x):
        if not isinstance(x,numbers.Number):
            raise TypeError('Unsupported Operand')
        return self.map(lambda e:e/x)

    def __floordiv__(self, x):
        if not isinstance(x,numbers.Number):
            raise TypeError('Unsupported Operand')
        return self.map(lambda e:e//x)

    def __mod__(self, x):
        if not isinstance(x,numbers.Number):
            raise TypeError('Unsupported Operand')
        return Matrix([j%x for j in i] for i in self)

    def __round__(self, n=None):
        return Matrix([round(j,n) for j in i] for i in self)

    def __pow__(self,x,m=None):
        """
        Matrix exponentiation\n
        Complexity: O(log2(x) * n^3)
        """
        if not isinstance(x,int):
            raise NotImplementedError
        if not self.is_square():
            raise DimensionError('Not square matrix')
        if m:
            return self.__modpow(x,m)
        a=self
        res=Matrix.identity(self.n)
        while x:
            if x&1: res@=a
            a*=a
            x>>=1
        return res

    def __modpow(self,p,m):
        a=self%m
        res=self.identity(self.n)
        while p:
            if p&1: res=(res*a)%m
            a=(a*a)%m
            p>>=1
        return res

    def __invert__(self):
        'inverse'
        return self.inverse()

    def __or__(self,other):
        """equivalent to concat()"""
        return self.concat(other)

    def __copy__(self):
        return Matrix(i for i in self)

    def __sizeof__(self):
        return self.__a.__sizeof__()

    def __repr__(self):
        return f'Matrix({repr(self.__a)})'

    def __str__(self):
        s=[]
        try: f='n'; f'{self[0][0]:n}'
        except Exception: f=''
        col_len=[max(len(f'{self[j][i]+0:{f}}') for j in range(self.n))+1 for i in range(self.m)]
        for i in range(self.n):
            l=[]
            for j in range(self.m):
                z=f'{self[i][j]+0:{f}}'
                x=col_len[j]-len(z)
                l.append(' '*(x//2)+z+' '*(x-x//2))
            s.append(' ['+''.join(l)[:-1]+']')
        s[0]='['+s[0][2:]
        return '['+'\n'.join(s)+']'

    def __eq__(self, other):
        if self is other:
            return True
        if self.n!=other.n or self.m!=other.m:
            return False
        for i in range(self.n):
            for j in range(self.m):
                if self[i][j]!=other[i][j]:
                    return False
        return True

