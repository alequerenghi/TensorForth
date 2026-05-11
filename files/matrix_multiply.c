#include "matrix_multiply.h"

void simple_multiply(float * A, float * B, float * C, int n, int m, int p)
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
			float sum = 0.0f;
      for (int k = 0; k < p; k++) {
				sum += A[i * p + k] * B[k * m + j];
      }
			C[i * m + j] += sum;
    }
  }
}

void transposed_multiply(float * A, float * B, float * C, int n)
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
	C[i * n + j] += A[i * n + k] * B[j * n + k];
      }
    }
  }
}

void kernel(float * A, float * B, float * C, int x, int dx, int y, int dy, int z, int dz, int n, int m, int p)
{
  int mx = (x + dx > n) ? n : x + dx;
  int my = (y + dy > m) ? m : y + dy;
  int mz = (z + dz > p) ? p : z + dz;
  for (int i = x; i < mx; i++) {
    for (int j = y; j < my; j++) {
			// hardware register acceleration
			float sum = 0.0f;
      for (int k = z; k < mz; k++) {
				sum += A[i * p + k] * B[k * m + j];
      }
			C[i * m + j] += sum;
    }
  }
}

void blocked_multiply(float * A, float * B, float * C, int n, int m, int p)
{
  const int s1 = 16;
  const int s2 = 16;
  const int s3 = 16;

  for (int i = 0; i < n; i += s1) {
    for (int j = 0; j < m; j += s2) {
      for (int k = 0; k < p; k += s3) {
				kernel(A, B, C, i, s1, j, s2, k, s3, n, m, p);
      }
    }
  }
}
