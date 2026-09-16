import json
import pathlib
import pickle
import tempfile
import unittest
from unittest import mock

import prepare_kl_validation as preparation


class KLContinuationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.parent = pathlib.Path(temporary.name) / 'parent'
        self.target = self.parent.parent / 'fixed'
        source = self.parent / 'source'
        source.mkdir(parents=True)
        repository = pathlib.Path(preparation.__file__).resolve().parent
        (source / 'dreamer.py').write_bytes((repository / 'dreamer.py').read_bytes())
        (source / 'models.py').write_bytes(b'# old KL fixture')
        batch = dict(status='completed', source_sha256={
            name: preparation.digest(source / name) for name in ('dreamer.py', 'models.py')}, runs=[])
        manifest = {}
        for seed, step in enumerate((100050, 100094, 100004)):
            run = self.parent / f'seed_{seed}'
            (run / 'episodes' / 'demonstrations').mkdir(parents=True)
            (run / 'episodes' / f'online-{step + 1}.npz').write_bytes(b'replay')
            for demo in range(64):
                (run / 'episodes' / 'demonstrations' / f'demo-{demo}.npz').write_bytes(b'demo')
            (run / 'variables.pkl').write_bytes(pickle.dumps(dict(
                step=step, warmup=dict(model=1000, actor=3000), variables=[1.0])))
            for name in ('metrics.jsonl', 'run_metadata.jsonl'):
                (run / name).write_text('{}\n')
            manifest[f'seed_{seed}/variables.pkl'] = dict(sha256=preparation.digest(run / 'variables.pkl'))
            batch['runs'].append(dict(seed=seed, final_step=step, command=[
                'bash', 'run_wsl.sh', '-u', 'dreamer.py', '--seed', str(seed),
                '--steps', '100000', '--logdir', str(run)]))
        preparation.write_json(self.parent / 'batch_status.json', batch)
        preparation.write_json(self.parent / 'stage_status.json', dict(status='completed'))
        preparation.write_json(self.parent / 'checkpoint_manifest.json', manifest)

    def test_selected_seeds_preserved_and_queued_in_order(self):
        before = {str(p.relative_to(self.parent)): preparation.digest(p)
                  for p in self.parent.rglob('*') if p.is_file()}
        with mock.patch.object(preparation, 'require_free_space'):
            result = preparation.prepare(self.parent, self.target, [0, 2])
        batch = json.loads((self.target / 'batch_status.json').read_text())
        self.assertEqual([r['seed'] for r in batch['runs']], [0, 2])
        self.assertFalse((self.target / 'seed_1').exists())
        self.assertFalse((self.target / 'stage_status.json').exists())
        for entry, parent in zip(batch['runs'], result['parent_checkpoints']):
            run = pathlib.Path(entry['outdir'])
            self.assertEqual(entry['status'], 'pending')
            self.assertEqual(entry['command'][entry['command'].index('--steps') + 1], '300000')
            self.assertEqual(preparation.digest(run / 'variables.pkl'), parent['checkpoint_sha256'])
            self.assertTrue((run / 'checkpoints' / f"step_{parent['step']:09d}.pkl").exists())
        for name, checksum in before.items():
            self.assertEqual(preparation.digest(self.parent / name), checksum)

    def test_default_remains_seed_one(self):
        with mock.patch.object(preparation, 'require_free_space'):
            result = preparation.prepare(self.parent, self.target)
        self.assertEqual(result['seeds'], (1,))

    def test_invalid_seeds_rejected_before_writes(self):
        for seeds in ([], [0, 0], [3]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                preparation.prepare(self.parent, self.target, seeds)
        self.assertFalse(self.target.exists())

    def test_corrupt_second_parent_rejected_before_copy(self):
        (self.parent / 'seed_2' / 'variables.pkl').write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'hash'):
            preparation.prepare(self.parent, self.target, [0, 2])
        self.assertFalse(self.target.exists())

    def test_existing_destination_rejected(self):
        self.target.mkdir()
        with self.assertRaises(FileExistsError):
            preparation.prepare(self.parent, self.target, [0, 2])


if __name__ == '__main__':
    unittest.main()
