use core::fmt;
use std::ops::{Add, Neg, Sub, AddAssign, SubAssign, Div, DivAssign, Mul, MulAssign};
use num_traits::{Float, NumAssignOps, NumCast};

const EPSILON: f64 = 1e-6;

#[derive(Debug, Copy, Clone, Eq)]
pub struct Complex<T> 
where T: Float + NumAssignOps {
    pub re: T,
    pub im: T,
}

impl<T> From<T> for Complex<T>
where T: Float + NumAssignOps {
    fn from(value: T) -> Self {
        Complex {
            re: value,
            im: T::zero(),
        }
    }
}

impl<T> PartialEq for Complex<T>
where T: Float + NumAssignOps {
    fn eq(&self, other: &Self) -> bool {
        let epsilon = T::from(EPSILON).unwrap();
        return (self.re - other.re).abs() <= epsilon && (self.im - other.im).abs() <= epsilon
    }

    fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
    }
}


impl<T, U> PartialEq<U> for Complex<T>
where T: Float + NumAssignOps,
U: NumCast {
    fn eq(&self, other: &U) -> bool {
        self.re.to_f64().unwrap() == (other.to_f64().unwrap()) && self.im.is_zero()
    }

    fn ne(&self, other: &U) -> bool {
        !self.eq(other)
    }
}


impl<T> Neg for Complex<T>
where T: Float + NumAssignOps {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl<T> Add for Complex<T>
where T: Float + NumAssignOps {
    type Output = Self;
    
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl<T> AddAssign for Complex<T>
where T: Float + NumAssignOps {
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl<T> Sub for Complex<T>
where T: Float + NumAssignOps {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self + -rhs
    }
}

impl<T> SubAssign for Complex<T>
where T: Float + NumAssignOps {
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

impl<T> Mul for Complex<T>
where T: Float + NumAssignOps {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}


// Multiply by scalar.
impl<T> Mul<T> for Complex<T>
where T: Float + NumAssignOps {
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

impl<T> MulAssign for Complex<T>
where T: Float + NumAssignOps {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

// Multiply by scalar.
impl<T> MulAssign<T> for Complex<T>
where T: Float + NumAssignOps {
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs;
    }
}

impl<T> Div for Complex<T> 
where T: Float + NumAssignOps {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.conjugate() * (T::one() / (rhs.re * rhs.re + rhs.im * rhs.im))
    }
}

// Divide by scalar.
impl<T> Div<T> for Complex<T>
where T: Float + NumAssignOps {
    type Output = Self;

    fn div(self, rhs: T) -> Self {
        Self {
            re: self.re / rhs,
            im: self.im / rhs,
        }
    }
}

impl<T> DivAssign for Complex<T>
where T: Float + NumAssignOps {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

// Divide by scalar.
impl<T> DivAssign<T> for Complex<T>
where T: Float + NumAssignOps {
    fn div_assign(&mut self, rhs: T) {
        *self = *self / rhs;
    }
}

impl<T> Complex<T>
where T: Float + NumAssignOps {
    pub fn new(re: T, im: T) -> Self {
        Self { re, im }
    }

    pub fn one() -> Self {
        Self::new(T::one(), T::zero())
    }

    pub fn i() -> Self {
        Self::new(T::zero(), T::one())
    }

    pub fn norm_squared(&self) -> T {
        self.re * self.re + self.im * self.im
    }

    pub fn norm(&self) -> T {
        self.norm_squared().sqrt()
    }

    pub fn conjugate(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    pub fn pow(&self, n: i32) -> Self {
        let r = self.norm();
        let arg = self.im.atan2(self.re);
        let r_pow = r.powi(n.abs());
        let cos = (arg * (T::from(n).unwrap())).cos();  // cos(n * arg)
        let sin = (arg * (T::from(n).unwrap())).sin(); // sin(n * arg)
        let complex = Self::new(r_pow * cos, r_pow * sin);
        if n >= 0 {
            return complex;
        }
        Self::one() / complex
    }

    pub fn approximately_equal(&self, other: &Self, epsilon: Option<T>) -> bool {
        let epsilon = epsilon.unwrap_or(T::epsilon());
        return (self.re - other.re).abs() <= epsilon && (self.im - other.im).abs() <= epsilon
    }
}

impl<T> fmt::Display for Complex<T>
where T: Float + NumAssignOps + fmt::Display {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sign = match self.im.is_sign_negative() {
            true => '-',
            false => '+'
        };
        write!(f, "{} {} {}i", self.re, sign, self.im.abs())
    }
}






#[cfg(test)]
mod tests {
    use crate::Complex;

    #[test]
    fn from_real() {
        let a = Complex::from(3.0); 
        let b = Complex::from(3.0f32);
    }


    macro_rules! partial_eq_scalar {
        ($b:expr, $c:expr) => {
            let a = Complex::new(3.0, 0.0);
            assert!(!(a == $b));
            assert!(a != $b);
            assert!(a == $c);
            assert!(!(a != $c));

            let d = Complex::new(3.0, 1.0);
            assert!(!(d == $b));
            assert!(d != $b);
            assert!(!(d == $c));
            assert!(d != $c);
        };
    }

    #[test]
    fn partial_eq() {
        partial_eq_scalar!(-3.0, 3.0);
        partial_eq_scalar!(-3, 3);
    }
    
    #[test]
    fn neg() {
        let a = Complex::new(3.0, 5.0);
        let b = Complex::new(-3.0, -5.0);
        assert_eq!(-a, b);
    }

    #[test]
    fn add() {
        let a = Complex::new(3.0, 5.0);
        let b = Complex::new(-1.0, -2.0);
        let c = Complex::new(2.0, 3.0);
        assert_eq!(a + b, c);
    }

    #[test]
    fn add_assign() {
        let mut a = Complex::new(3.0, 5.0);
        let b = Complex::new(-1.0, -2.0);
        a += b;
        let c = Complex::new(2.0, 3.0);
        assert_eq!(a, c);
    }

    #[test]
    fn sub() {
        let a = Complex::new(3.0, 5.0);
        let b = Complex::new(-1.0, -2.0);
        let c = Complex::new(4.0, 7.0);
        assert_eq!(a - b, c);
    }

    #[test]
    fn sub_assign() {
        let mut a = Complex::new(3.0, 5.0);
        let b = Complex::new(-1.0, -2.0);
        a -= b;
        let c = Complex::new(4.0, 7.0);
        assert_eq!(a, c);
    }


    #[test]
    fn mul() {
        let a = Complex::new(3.0, 5.0);
        let b = Complex::new(-1.0, -2.0);
        let c = Complex::new(7.0, -11.0);
        assert_eq!(a * b, c);
    }

    #[test]
    fn mul_assign() {
        let mut a = Complex::new(3.0, 5.0);
        let b = Complex::new(-1.0, -2.0);
        a *= b;
        let c = Complex::new(7.0, -11.0);
        assert_eq!(a, c);
    }

    #[test]
    fn scale() {
        let a = Complex::new(3.0, 5.0);
        let f = 2.0;
        let a_mul_f = Complex::new(6.0, 10.0);
        let a_div_f = Complex::new(1.5, 2.5);
        assert_eq!(a * f, a_mul_f);
        assert_eq!(a / f, a_div_f);
    }

    #[test]
    fn scale_assign() {
        let a = Complex::new(3.0, 5.0);
        let mut b = Complex::new(3.0, 5.0);
        let f = 2.0;
        b *= f;
        let c = Complex::new(6.0, 10.0);
        assert_eq!(b, c);
        b /= f;
        assert_eq!(b, a);
    }

    #[test]
    fn div() {
        let a = Complex::new(3.0, 5.0);
        let b = Complex::new(-1.0, -2.0);
        let c = Complex::new(-13.0 / 5.0, 1.0 / 5.0);
        assert_eq!(a / b, c);
    }

    #[test]
    fn div_assign() {
        let mut a = Complex::new(3.0, 5.0);
        let b = Complex::new(-1.0, -2.0);
        a *= b;
        let c = Complex::new(7.0, -11.0);
        assert_eq!(a, c);
    }

    #[test]
    fn norm() {
        assert_eq!(Complex::new(3.0, 4.0).norm(), 5.0);
        assert_eq!(Complex::new(2.0, 0.0).norm_squared(), 4.0);
    }

    #[test]
    fn pow() {
        let a = Complex::new(3.0, 3.0);
        let a_0: Complex<f64> = Complex::one();
        let a_1 = a.clone();
        let a_5 = Complex::new(-972.0, -972.0);
        assert_eq!(a.pow(0), a_0);
        assert_eq!(a.pow(1), a_1);
        assert_eq!(a.pow(5), a_5);
        assert!(a.pow(5).approximately_equal(&a_5, Some(0.001)));
    }
}









fn main() {
    let a = Complex::new(1.0, -5.3);
    println!("{}", a);
    let a = Complex::new(0.0, 2.3);
    println!("{}", a);
}