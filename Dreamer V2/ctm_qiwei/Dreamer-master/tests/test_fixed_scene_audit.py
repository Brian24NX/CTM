import unittest
import numpy as np
from audit_fixed_scenes import unstall, paired, prediction_errors


class FixedSceneAuditTest(unittest.TestCase):
    def test_turn_guard_uses_fifth_zero_speed_turn_and_keeps_original(self):
        original = np.array([0.,1.,0.,.4],np.float32)
        streak = 0
        for i in range(5):
            action, streak, triggered = unstall(original,-.5,0.,streak)
            self.assertEqual(triggered,i==4)
        np.testing.assert_allclose(action,[1.,0.,-.5,0.])
        np.testing.assert_allclose(original,[0.,1.,0.,.4])
        action, _, hit = unstall(original,-.5,0.,4,positive=True)
        np.testing.assert_allclose(action,[1.,0.,1.,0.])
        self.assertTrue(hit)

    def test_moving_or_move_resets_turn_guard(self):
        for action,speed in (([0.,1.,0.,.2],2.),([1.,0.,-.5,0.],0.)):
            actual, streak, hit = unstall(action,.5,speed,4)
            self.assertEqual(streak,0)
            self.assertFalse(hit)
            np.testing.assert_allclose(action,actual)

    def test_paired_counts_and_scene_alignment(self):
        a=[dict(scene=i,success=False,pickup=False,timeout=True,oob=False) for i in range(4)]
        b=[dict(r,success=i<2,pickup=i<3,timeout=i>=2) for i,r in enumerate(a)]
        result=paired(a,b)
        self.assertEqual(result['success']['delta_pp'],50.)
        self.assertEqual(result['success']['gained'],2)
        self.assertEqual(result['success']['lost'],0)
        with self.assertRaises(AssertionError):paired(a,list(reversed(b)))

    def test_physical_units_and_terminal_mask(self):
        true=np.zeros((2,13),np.float32); true[:,4]=1
        pred=true.copy(); pred[:,0]=.1; pred[:,11]=.1; pred[:,2]=.1; pred[:,9]=.1
        report=prediction_errors(pred,true,true,np.zeros(2),np.array([1,0]),
                                 np.array([0.,1.]),np.array([.99,.1]))
        self.assertAlmostEqual(report['position_mae_m'],100.,places=3)
        self.assertAlmostEqual(report['active_goal_mae_m'],200.,places=3)
        self.assertAlmostEqual(report['speed_mae'],2.,places=3)
        self.assertAlmostEqual(report['time_mae_steps'],5.,places=3)
        self.assertEqual(report['false_terminal_fraction'],0.)


if __name__=='__main__':unittest.main()
