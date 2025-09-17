"""
    Copyright © Advanced Micro Devices, Inc., or its affiliates
    SPDX-License-Identifier:  MIT
"""
"""
    TEST DESCRIPTION: This test build and run the MSCCL_rccltests tests.
"""

from execution_APIs.test import test
from logParser.tests.parse_pass_fail_re import parse_pass_fail_re
from DB.db import RCCLUTTestResult as DBRCCLUTTestResult
from execution_APIs.input import Input
from testResultParser.post_execution import *
from platforms.rocm.get_rocm_utils import get_rocm_utils, get_rocm_utils_ip, rocm_utils
from logParser.tests.prerequisites_failed import is_prerequisite_failed_in_testcase
from platforms.rocm.get_rocm_utils import get_rocm_utils, get_rocm_utils_ip
import logging, os


class MscclRcclTests():
    """!
    This class build and run the MSCCL_rccltests tests.

    """

    def __init__(self, data=None):

        self.logger = logging.getLogger(__name__)
        self.rocm_utils_objects = get_rocm_utils(rocm_utils_ip=get_rocm_utils_ip())
        self.platform = self.rocm_utils_objects.platform
        self.rocm_dir = self.rocm_utils_objects.rocm_utils.get_rocm_dir()
        self.platform_utils = self.rocm_utils_objects.platform_utils

    def get_rccl_source(self):

        """!
        This method git clone the rccl repo.
        @param: None
        @return: home_dir rccl downloaded  path
        """

        gitop = GitOp(platform=self.platform)
        home_dir = os.path.expanduser('~')
        git_data = {}
        if self.git_branch == "main":
            git_data["git_repo"] = {"location": home_dir,  "download_url": "https://github.com/ROCm/rccl-tests",
                                    "reponame": "rccl-tests"}
        else:
            git_data["git_repo"] = {"location": home_dir,"branch": self.git_branch, "download_url": "https://github.com/ROCm/rccl-tests",
                                    "reponame": "rccl-tests"}
        gitclone_ret = gitop.execute(data=git_data)
        home_dir = os.path.join(home_dir, 'rccl-tests')
        return home_dir

    def execute(self, executeData: Input):

        """!This method will execute MSCCL_rccltests test script and validate the test result.
        @param: executeData (Input): Input data
        @return: Dictionary conatains test result status
        """
        test_result = None
        self.git_branch  = executeData.rocm_test_executor.git_branch
        cmd_dir = self.get_rccl_source()
        if os.path.exists(cmd_dir):
            cmd_dir = cmd_dir+"/build"
            self.platform_utils.mkdir(cmd_dir)
            tester_op = test(cmd=f"CXX={self.rocm_dir}/bin/hipcc cmake -DCMAKE_PREFIX_PATH={self.rocm_dir}/lib ..", cmd_dir=cmd_dir)
            tester_op = test(cmd="make", cmd_dir=cmd_dir)
            self.test_suit_name = 'msccl_rccltests'
            if os.path.exists(cmd_dir):
                exe_files_list= ['all_reduce_perf', 'alltoall_perf']
                match_logs = {'all_reduce_perf': 'mscclFuncAllReduce', 'alltoall_perf': 'mscclFuncAllToAll'}
                for exe in exe_files_list:
                    tester_op = test(cmd=f"HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL ./{exe} -b 1M -e 10M -f 2 -g 1 -t 8 -o sum -d float -n 1", cmd_dir=cmd_dir, test_log_parser_fn=parse_pass_fail_re, test_case_name=f"{exe}", test_log_parser_args=({"PASS": {"match_re": [match_logs[exe]]} },),db_test_name='msccl_rccltests', test_db=DBRCCLUTTestResult(), test_result=test_result,test_suite_name="msccl_rccltests")
                    test_result = tester_op.test_result

            else:
                is_prerequisite_failed_in_testcase(tester_op.test_result,db_obj=DBRCCLUTTestResult(), test_name='msccl_rccltests')
        else:

            self.logger.error(f'{cmd_dir} path not existed!!')
            is_prerequisite_failed_in_testcase(test_result, db_obj=DBRCCLUTTestResult(), test_name='msccl_rccltests')
        return {
            "msccl_rccltests": tester_op.test_result
        }
