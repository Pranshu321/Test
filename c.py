"""
    Copyright © Advanced Micro Devices, Inc., or its affiliates
    SPDX-License-Identifier:  MIT
"""
"""
    TEST DESCRIPTION: This test build and run the MSCCL_rccltests tests.
"""

from execution_APIs.test import test
from NDA.check import parse_pass_fail_re

class MscclRcclTests():
    """!
    This class build and run the MSCCL_rccltests tests.

    """

    def get_rccl_source(self):

        """!
        This method git clone the rccl repo.
        @param: None
        @return: home_dir rccl downloaded  path
        """
