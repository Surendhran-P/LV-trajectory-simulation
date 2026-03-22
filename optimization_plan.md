# Three-Stage Rocket Mass Optimization Plan

Goal: Build a Python program that minimizes total lift-off mass for a fixed three-stage rocket using an explicit Lagrange multiplier formulation with NumPy only.

## Problem Setup

- Fixed stage count: 3
- Given per stage inputs:
  - Specific impulse (Isp)
  - Structural mass fraction
- Mission constraints:
  - Required total delta-v
  - Required payload mass
- Outputs:
  - Optimal stage mass ratios
  - Minimum lift-off mass
  - Per-stage propellant and structure masses

## Implementation Steps

1. Define model variables in staging.py
- Represent decision variables as stage mass ratios for each stage.
- Store mission inputs and stage properties.

2. Build governing equations
- Use the ideal rocket equation to express stage delta-v contributions.
- Build total delta-v equality constraint.
- Build payload-linked mass recursion across stages.

3. Form Lagrangian
- Objective: minimize total initial mass.
- Add equality constraints with Lagrange multipliers.
- Derive residual system consisting of:
  - Stationarity equations (partial derivatives set to zero)
  - Constraint equations

4. Implement numerical solver (NumPy only)
- Solve residual system with Newton-Raphson.
- Compute Jacobian using finite differences.
- Add damping or line-search style step scaling for stability.
- Add convergence criteria and iteration limits.

5. Reconstruct physical masses
- Compute per-stage wet mass, dry mass, structure mass, and propellant mass.
- Compute total lift-off mass.
- Verify positivity and physical feasibility.

6. Build executable interface
- Add a main block to run a nominal case.
- Print readable report:
  - Stage mass ratios
  - Total mass
  - Stage-wise mass breakdown
  - Constraint residuals

7. Verification
- Nominal case should converge with low residuals.
- Increase required delta-v and verify total mass increases.
- Reduce one stage Isp and verify total mass increases.
- Run one infeasible case and verify clear failure message.

## File Responsibilities

- staging.py
  - Core equations
  - Lagrangian residual formulation
  - Nonlinear solver
  - Mass reconstruction and reporting

- trajectory.py
  - Keep unchanged unless a small utility extraction improves readability.

## Acceptance Criteria

- Program runs from command line with Python.
- Numerical solution converges for nominal inputs.
- Residuals for constraints and stationarity are near zero.
- Output includes all requested optimization results.
