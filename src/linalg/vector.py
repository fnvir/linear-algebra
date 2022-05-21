import numbers
import math

__all__=['Vector']


class Vector:
    """3D vectors"""
    def __init__(self,_i=0,_j=0,_k=0,fractional=False):
        if fractional:
            from fractions import Fraction
            _i,_j,_k=map(Fraction,[_i,_j,_k])
        self.__v=(_i,_j,_k)

    def dot(self,o):
        return sum(self[i]*o[i] for i in range(3))

    def cross(s,o):
        return Vector(s[1]*o[2]-s[2]*o[1],s[2]*o[0]-s[0]*o[2],s[0]*o[1]-s[1]*o[0])

    @staticmethod
    def angle(a,b,radian=False):
        """Angle between two vectors a and b"""
        x=math.acos((a*b)/(abs(a)*abs(b)))
        return x if radian else math.degrees(x)

    def project_on(a,b):
        return (a.dot(b)/sum(i*i for i in b))*b

    def unit(self):
        return self/self.norm()

    def cardinality(self):
        return sum(i!=0 for i in self)

    @property
    def dim(self):
        return self.cardinality()

    def norm(self):
        return sum(i*i for i in self)**.5

    def map(self,f):
        return Vector(*(f(x) for x in self))

    def __len__(self):
        return 3

    def __getitem__(self,i):
        return self.__v[i]

    def __copy__(self):
        return Vector(*self)

    def __abs__(self):
        return self.norm()

    def __pos__(self):
        return self

    def __neg__(self):
        return Vector(-self[0],-self[1],-self[2])

    def __add__(a,b):
        if not isinstance(b,Vector):
            raise TypeError(f'Unsupported operand for {type(a)} and {type(b)}')
        return Vector(*[a[i]+b[i] for i in range(3)])

    def __radd__(a,b):
        return a+b

    def __sub__(a,b):
        return a+(-b)

    def __rsub__(a,b):
        return a-b

    def __mul__(a,b):
        if isinstance(b,Vector):
            return a.dot(b)
        if not isinstance(b,numbers.Real):
            raise TypeError(f'Unsupported operand for {type(a)} and {type(b)}')
        return Vector(*[i*b for i in a])

    def __rmul__(a,b):
        return a*b

    def __matmul__(a,b):
        if not isinstance(b,Vector):
            raise TypeError(f'Unsupported operand for {type(a)} and {type(b)}')
        return a.cross(b)

    def __truediv__(a,b):
        if not isinstance(b,numbers.Real):
            raise TypeError(f'Unsupported operand for {type(a)} and {type(b)}')
        return Vector(*(i/b for i in a))

    def __floordiv__(a,b):
        if not isinstance(b,numbers.Real):
            raise TypeError(f'Unsupported operand for {type(a)} and {type(b)}')
        return Vector(*[i//b for i in a])

    def __mod__(a,b):
        if not isinstance(b,numbers.Integral):
            raise TypeError(f'Unsupported operand for {type(a)} and {type(b)}')
        return Vector(*(i%b for i in a))

    def __pow__(self,p,m=None):
        if m:
            return self.__modpow(p,m)
        a=self
        res=1
        while p:
            if p&1: res*=a
            a*=a
            p>>=1
        return res

    def __modpow(self,p,m):
        a=self%m
        res=1
        while p:
            if p&1: res=(res*a)%m
            a=(a*a)%m
            p>>=1
        return res

    def __trunc__(self):
        return Vector(*[i.__trunc__() for i in self])

    def __floor__(self):
        return Vector(*[i.__floor__() for i in self])

    def __ceil__(self):
        return Vector(*[i.__ceil__() for i in self])

    def __round__(self,n=None):
        return Vector(*[round(i,n) for i in self])

    def __bool__(self):
        return self.cardinality()

    def __sizeof__(self):
        return self.__v.__sizeof__()

    def __eq__(self,other):
        return all(self[i]==other[i] for i in range(3))

    def __str__(self):
        return f"[{' '.join(map(str,self.__v))}]"

    def __repr__(self):
        return 'Vector({}, {}, {})'.format(*self)
