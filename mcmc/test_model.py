import unittest
import model
import numpy.testing as npt

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


if __name__ == '__main__':
    unittest.main()
