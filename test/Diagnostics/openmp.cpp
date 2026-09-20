#include "clad/Differentiator/Differentiator.h"
#include <cstdio>
#include <cmath>
#include <omp.h>

// Funkcja testowa z odczytem sąsiada (stencil)
double fn(const double* x, int n) {
  double total = 0.0;
  #pragma omp parallel for reduction(+:total)
  for (int i = 1; i < n; i++) {
    total += x[i] * x[i - 1];
  }
  return total;
}

// Analityczne obliczenie oczekiwanego gradientu dla x[i] = i * 0.5
// dla f(x) = x[1]*x[0] + x[2]*x[1] + ... + x[n-1]*x[n-2]
// df/dx[0] = x[1]
// df/dx[i] = x[i-1] + x[i+1] (dla 0 < i < n-1)
// df/dx[n-1] = x[n-2]
double compute_expected_sum(int n) {
  double expected_sum = 0.0;
  for (int i = 0; i < n; ++i) {
    double grad_i = 0.0;
    if (i > 0)     grad_i += (i - 1) * 0.5; // od pochodnej czlonu x[i]*x[i-1] po x[i]
    if (i < n - 1) grad_i += (i + 1) * 0.5; // od pochodnej czlonu x[i+1]*x[i] po x[i]
    expected_sum += grad_i;
  }
  return expected_sum;
}

int main() {
  const int n = 64;
  const int num_runs = 50; // Liczba powtórzeń dla wyłapania niedeterminizmu
  double x[n], d[n];

  // Inicjalizacja danych
  for (int i = 0; i < n; ++i) x[i] = i * 0.5;

  // Generowanie gradientu przez Clad
  auto g = clad::gradient(fn, "x");

  double expected = compute_expected_sum(n);
  printf("=== Wartość oczekiwana (analitycznie): %.6f ===\n\n", expected);

  // Testowanie dla różnej liczby wątków OpenMP
  int thread_counts[] = {1, 2, 4, 8};

  for (int threads : thread_counts) {
    omp_set_num_threads(threads);
    int error_count = 0;
    double min_val = 1e9, max_val = -1e9;

    for (int run = 0; run < num_runs; ++run) {
      for (int i = 0; i < n; ++i) d[i] = 0.0;

      g.execute(x, n, d);

      double sum = 0.0;
      for (int i = 0; i < n; ++i) sum += d[i];

      if (std::abs(sum - expected) > 1e-4) {
        error_count++;
      }

      if (sum < min_val) min_val = sum;
      if (sum > max_val) max_val = sum;
    }

    printf("Wątki: %d | Błędne przebiegi: %2d/%d | Min suma: %.6f | Max suma: %.6f | Stan: %s\n",
           threads,
           error_count,
           num_runs,
           min_val,
           max_val,
           (error_count == 0 ? "OK" : "BŁĄD (Data Race)"));
  }

  return 0;
}