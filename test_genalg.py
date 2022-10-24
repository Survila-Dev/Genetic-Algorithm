"""
Testing the genetical algorithm modul.

Author: Eimantas Survila
Date: Sat Dec 19 08:30:42 2020
"""

import unittest
import genalg_def as genalg
import symbolicregression as symreg

import numpy as np


class GenAlgTest(unittest.TestCase):

    # %%
    def test_circle_func(self):
        """
        Test if float variable optimization for circle functions works.
        """

        def func_to_optimize_1(a, b):
            return (a**2 + b**2)**0.5

        vartype = {
            "x": float,
            "y": float}

        varbounds = {
            "x": [-50, 50],
            "y": [-50, 50]}

        def func_in(gencode):
            return gencode["x"], gencode["y"]

        def func_fit(func_output):
            return abs(func_output)

        optimizer = genalg.Evolver(
            var_type=vartype,
            var_bounds=varbounds,
            f=func_to_optimize_1,
            f_in=func_in,
            f_fit=func_fit)

        optimizer.configuration["population_size"] = 100
        optimizer.configuration["maximal_iterations"] = 20

        optimizer.run()

        [a_best, b_best] = optimizer.bestvars()
        best_f_out = optimizer.bestoutput()

        self.assertAlmostEqual(a_best, 0, 0)
        self.assertAlmostEqual(b_best, 0, 0)
        self.assertAlmostEqual(best_f_out, 0, 0)

    # %%
    def test_discrete_vals(self):
        """
        Test if optimization with discrete variabels from a list works.
        """

        def func_to_optimize_2(a, b):

            if a == "Vorname" and b == "Nachname":
                return 0.1
            else:
                return 0.5

        vartype = {
            "first_name": list,
            "last_name": list}

        varbounds = {
            "first_name": ["Name", "Titel", "Nachname", "Vorname"],
            "last_name": ["Vorname", "Doktor", "Professor", "Nachname"]}

        def func_in(gencode):
            return gencode["first_name"], gencode["last_name"]

        def func_fit(func_output):
            return func_output

        optimizer = genalg.Evolver(
            var_type=vartype,
            var_bounds=varbounds,
            f=func_to_optimize_2,
            f_in=func_in,
            f_fit=func_fit)

        optimizer.configuration["population_size"] = 50
        optimizer.configuration["maximal_iterations"] = 10

        optimizer.run()

        [first_best, last_best] = optimizer.bestvars()
        best_f_out = optimizer.bestoutput()

        self.assertEqual(first_best, "Vorname")
        self.assertEqual(last_best, "Nachname")
        self.assertEqual(best_f_out, 0.1)

    # %%
    def test_mixed_variables(self):
        """
        Test if optimization with discrete values works.
        """

        def func_to_optimize_3(no_bricks, brick_type, wall_height):

            if brick_type == "brick1":
                brick_len = 10
            if brick_type == "brick2":
                brick_len = 12
            if brick_type == "brick3":
                brick_len = 8

            return 0, 0, brick_len * float(no_bricks) * wall_height

        vartype = {
            "no_bricks": int,
            "brick_type": list,
            "wall_height": float}

        varbounds = {
            "no_bricks": [1, 21],  # The last value is not inclusive.
            "brick_type": ["brick1", "brick2", "brick3"],
            "wall_height": [1, 0.2, 10]}

        def func_in(gencode):
            return gencode["no_bricks"], gencode["brick_type"], gencode["wall_height"]

        def func_fit(func_output):
            return 1/func_output[2]

        optimizer = genalg.Evolver(
            var_type=vartype,
            var_bounds=varbounds,
            f=func_to_optimize_3,
            f_in=func_in,
            f_fit=func_fit)

        optimizer.configuration["population_size"] = 150
        optimizer.configuration["maximal_iterations"] = 80

        optimizer.run()

        [best_int, best_brick, best_wall_h] = optimizer.bestvars()
        [out1, out2, out3] = optimizer.bestoutput()

        self.assertEqual(best_int, 20)
        self.assertAlmostEqual(best_wall_h, 10, 1)
        self.assertEqual(best_brick, "brick2")
        self.assertEqual(out1, 0)
        self.assertEqual(out2, 0)
        self.assertAlmostEqual(out3, 2400, 0)

    # %%
    def test_bifurcating_arguments(self):
        """
        Test if bifurcating arguments are handled right.
        """

        def func_to_optimize_4(subfunc_type, *args):

            def square(*args):
                if len(args) != 2:
                    raise "Too many variables"
                return args[0] * args[1]

            def cube(*args):
                return args[0] * args[1] * args[2]

            if subfunc_type == "Square":

                return square(*args) * 20

            if subfunc_type == "Cube":

                return cube(*args)

        vartype = {
            "Subfunc_type": list,
            "length": float,
            "width": float,
            "height": float}

        varbounds = {
            "Subfunc_type": ["Square", "Cube"],
            "length": [1, 20],
            "width": [1, 19],
            "height": [1, 21]}

        def func_in(gencode):

            if gencode["Subfunc_type"] == "Square":

                return \
                    gencode["Subfunc_type"], \
                    gencode["length"], \
                    gencode["width"]

            if gencode["Subfunc_type"] == "Cube":

                return \
                    gencode["Subfunc_type"], \
                    gencode["length"], \
                    gencode["width"], \
                    gencode["height"]

        def func_fit(func_output):
            return 1/func_output

        optimizer = genalg.Evolver(
            var_type=vartype,
            var_bounds=varbounds,
            f=func_to_optimize_4,
            f_in=func_in,
            f_fit=func_fit)

        optimizer.configuration["population_size"] = 200
        optimizer.configuration["maximal_iterations"] = 50

        optimizer.run()

        best_vars = optimizer.bestvars()
        if len(best_vars) == 3:
            out_form, out_l, out_w = best_vars
        if len(best_vars) == 4:
            out_form, out_l, out_w, out_h = best_vars

        best_f_out = optimizer.bestoutput()
        if out_form == "Cube":
            self.assertAlmostEqual(best_f_out, 7980, 0)
            self.assertEqual(out_form, "Cube")
            self.assertAlmostEqual(out_l, 20, 0)
            self.assertAlmostEqual(out_w, 19, 0)
            self.assertAlmostEqual(out_h, 21, 0)
        if out_form == "Square":
            self.assertAlmostEqual(best_f_out, 7600, 0)
            self.assertEqual(out_form, "Square")
            self.assertAlmostEqual(out_l, 20, 0)
            self.assertAlmostEqual(out_w, 19, 0)

    # %%
    def test_int_boundaries(self):
        """
        Test if integers boundaries and step size is handled correctly.
        """

        def func_to_optimize_5(h, v):
            return abs((h-5)*v)

        vartype = {
            "horz": int,
            "vert": int}

        varbounds = {
            "horz": [2, 5],
            "vert": [6, 12]}

        def func_in(gencode):
            return gencode["horz"], gencode["vert"]

        def func_fit(func_output):
            return func_output

        optimizer = genalg.Evolver(
            var_type=vartype,
            var_bounds=varbounds,
            f=func_to_optimize_5,
            f_in=func_in,
            f_fit=func_fit)

        optimizer.configuration["population_size"] = 100
        optimizer.configuration["maximal_iterations"] = 50

        optimizer.run()

        [h_best, v_best] = optimizer.bestvars()
        best_f_out = optimizer.bestoutput()

        # print("h best is {} and v_best is {}".format(h_best, v_best))
        self.assertTrue(h_best == 4 or h_best == 6)
        self.assertAlmostEqual(v_best, 6)
        self.assertAlmostEqual(best_f_out, 6, 0)

    # %%
    def test_float_step_sizes(self):
        """
        Test if float variable step sizes are handled correctly.
        """

        def func_float_step_size(x, y):
            return abs((x - 0.75)*(y - 0.75))

        vartype = {
            "x": float,
            "y": float}

        varbounds = {
            "x": [-60, 0.5, 60],
            "y": [-70, 1/3, 70]}

        def func_in(gencode):
            return gencode["x"], gencode["y"]

        def func_fit(func_output):
            return func_output

        optimizer = genalg.Evolver(
            var_type=vartype,
            var_bounds=varbounds,
            f=func_float_step_size,
            f_in=func_in,
            f_fit=func_fit)

        optimizer.configuration["population_size"] = 100
        optimizer.configuration["maximal_iterations"] = 50
        optimizer.configuration["log_results"] = False

        optimizer.run()

        [x_best, y_best] = optimizer.bestvars()
        best_f_out = optimizer.bestoutput()

        self.assertTrue(x_best == 0.5 or x_best == 1)
        self.assertAlmostEqual(y_best, 2/3, 1)
        self.assertAlmostEqual(best_f_out, 0.0208, 2)


# %%
if __name__ == "__main__":
    unittest.main()
