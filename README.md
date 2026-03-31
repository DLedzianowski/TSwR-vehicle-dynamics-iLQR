# Vehicle Dynamics - iLQR
## CEL
* bicycle model + iLQR


### 1. Model bazowy ([commonroad-vehicle-models](https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models))


### 2. Model dynamiczny ([paper](https://arxiv.org/pdf/2003.04882))

### 3. Model opon (linear tire model/Pacejka)

### 4. Sterowanie

* regulator pure pursuit (do sprawdzania modelu)
* iLQR

## 5. Narzędzia i symulacja

* Python
* NumPy – obliczenia macierzowe
* SciPy – integracja równań (Euler / RK4 / solve_ivp)
* Matplotlib – wizualizacja (trajektoria, błędy, sterowania)
* [commonroad-vehicle-models](https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models) – model bazowy pojazdu
* [CasADi](https://github.com/casadi/casadi) – obliczanie pochodnych do iLQR

# Milestone

* zaimplementowany [bicycle model](https://arxiv.org/pdf/2003.04882):
  * kinematyczny i podstawowy dynamiczny
* działający liniowy model opon
* regulator Pure Pursuit:
  * stabilne śledzenie trajektorii
* wizualizacja:
  * tor jazdy
  * błąd śledzenia trajektorii
  * sygnały sterujące
