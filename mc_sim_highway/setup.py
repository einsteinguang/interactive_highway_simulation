from distutils.core import setup

setup (name = 'mc_sim_highway',
       version = '0.1',
       author = "Lingguang Wang",
       description = """Install precompiled extension""",
       py_modules = ["mc_sim_highway"],
       packages=[''],
       package_data={'': ['mc_sim_highway.so']},
       )
