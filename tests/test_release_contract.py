"""CPU-only release checks; these are not reconstruction-quality validation."""
from pathlib import Path
import ast
import runpy
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]

class ReleaseContract(unittest.TestCase):
    def test_first_party_syntax(self):
        for folder in ('VSLAM','Reconstruct','config','dataloaders'):
            for path in (ROOT/folder).rglob('*.py'):
                if {'thirdparty','submodules','__pycache__'} & set(path.parts):
                    continue
                ast.parse(path.read_text(),filename=str(path))
        ast.parse((ROOT/'run_system.py').read_text())

    def test_static_defaults(self):
        with patch.object(sys,'argv',['run_system.py','artdeco3d','-s','example','-m','results/test','--model_weight','example.pt']):
            args=runpy.run_path(str(ROOT/'config/args/artdeco3d.py'))['get_args']()
        self.assertFalse(args.use_dynamic)
        self.assertEqual(args.model_conf,'config/yaml/pi3/pi3match.yaml')

    def test_matching_configs(self):
        for name in ('pi3match','pi3xmatch'):
            text=(ROOT/f'config/yaml/pi3/{name}.yaml').read_text()
            self.assertIn(f'VSLAM.models.pi3.models.{name}',text)
            self.assertTrue((ROOT/f'VSLAM/models/pi3/models/{name}.py').is_file())

    def test_shell_help(self):
        p=subprocess.run(['bash','run.sh','--help'],cwd=ROOT,capture_output=True,text=True,check=True)
        self.assertIn('MATCHING_CHECKPOINT',p.stdout)

    def test_no_cluster_paths_first_party(self):
        for folder in ('VSLAM','Reconstruct','config','dataloaders'):
            for p in (ROOT/folder).rglob('*'):
                if p.suffix not in ('.py','.sh','.yaml') or {'thirdparty','submodules'} & set(p.parts):continue
                for marker in ('/cpfs/','/shared-data/','/home/renkerui','/root/'):
                    self.assertNotIn(marker,p.read_text(),str(p))

    def test_optimizer_api_sources(self):
        root=ROOT/'Reconstruct/submodules'
        text=(root/'diff-gaussian-rasterization/ext.cpp').read_text()
        self.assertIn('adamUpdateBasic',text)
        self.assertIn('distIndex2',(root/'simple-knn/ext.cpp').read_text())

if __name__=='__main__':unittest.main()
