import unittest
import model
import numpy.testing as npt
import mcmc

class TestModel(unittest.TestCase):
    def test_dual_band(self):
        params = model.EggParams()
        sb_img = model.egg(params, nx=50, ny=50)
        db1, db2 = model.egg(params, dual_band=True, match_image_size=sb_img)
        db_img = db1 + db2

        #troubleshooting
        #model.show(sb_img)
        #model.show(db_img, colorbar=True)
        #model.show(sb_img - db_img, colorbar=True)
        #print(sum(sum(sb_img.array)))
        #print(sum(sum(db_img.array)))

        npt.assert_array_almost_equal(db_img.array, sb_img.array)

    def test_dual_band_noise(self):
        params = model.EggParams()
        data1, var1 = mcmc.generate_data(params, dual_band=False)
        data2, var2 = mcmc.generate_data(params, dual_band=True)

        N = 50
        N2 = float(N**2)
        a1 = data1.array[:N][:N]
        a2 = (data2[0] + data2[1]).array[:N][:N]

        mean1 = sum(sum(a1))/N2
        mean2 = sum(sum(a2))/N2

        var1 = sum(sum((a1-mean1)**2))/N2
        var2 = sum(sum((a2-mean2)**2))/N2

        #model.show(data1, colorbar=True)
        #model.show(data2[0] + data2[0], colorbar=True)

        #assert fractional difference in variance is less than 5%
        self.assertLess(abs(var1-var2)/var1, 0.05)

if __name__ == '__main__':
    unittest.main()
