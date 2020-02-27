/*
 * S-function communicating with square wave signal generator via TCP
 *
 * Copyright (C) 2020 Martin Gurtner <martin.gurtner@fel.cvut.cz>
 * (Based on a Pavel Pisa's (pisa@cmp.felk.cvut.cz) S function example)
 *
 * Department of Control Engineering
 * Faculty of Electrical Engineering
 * Czech Technical University in Prague (CTU)
 *
 * The S-Function for can be distributed in compliance
 * with GNU General Public License (GPL) version 2 or later.
 * Other licence can negotiated with CTU.
 *
 * Next exception is granted in addition to GPL.
 * Instantiating or linking compiled version of this code
 * to produce an application image/executable, does not
 * by itself cause the resulting application image/executable
 * to be covered by the GNU General Public License.
 * This exception does not however invalidate any other reasons
 * why the executable file might be covered by the GNU Public License.
 * Publication of enhanced or derived S-function files is required
 * although.
 *
 * sfuntmpl_basic.c by The MathWorks, Inc. has been used to accomplish
 * required S-function structure.
 */


#define S_FUNCTION_NAME  sfTCPgenerator
#define S_FUNCTION_LEVEL 2

/*
 * The S-function has next parameters
 *
 * Sample time     - sample time value or -1 for inherited
 * Counter Mode    -
 * Counter Gating
 * Reset Control
 * Digital Filter
 */

#define PRM_TS(S)                   (mxGetScalar(ssGetSFcnParam(S, 0)))
#define PRM_IPADDR(S)               (ssGetSFcnParam(S, 1))
#define PRM_TCP_PORT(S)             (mxGetScalar(ssGetSFcnParam(S, 2)))

#define PRM_COUNT                   3

#define IWORK_IDX_TCP_FD            0

#define IWORK_COUNT                 1

#define IWORK_TCP_FD(S)             (ssGetIWork(S)[IWORK_IDX_TCP_FD])


/*
 * Need to include simstruc.h for the definition of the SimStruct and
 * its associated macro definitions.
 */
#define _POSIX_C_SOURCE 200112L

#include "simstruc.h"

#ifndef WITHOUT_HW 

#include <sys/types.h>
#include <sys/socket.h> 
#include <sys/stat.h>
#include <netdb.h> 
#include <stdio.h>
#include <fcntl.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>

#endif /*WITHOUT_HW*/

/* Error handling
 * --------------
 *
 * You should use the following technique to report errors encountered within
 * an S-function:
 *
 *       ssSetErrorStatus(S,"Error encountered due to ...");
 *       return;
 *
 * Note that the 2nd argument to ssSetErrorStatus must be persistent memory.
 * It cannot be a local variable. For example the following will cause
 * unpredictable errors:
 *
 *      mdlOutputs()
 *      {
 *         char msg[256];         {ILLEGAL: to fix use "static char msg[256];"}
 *         sprintf(msg,"Error due to %s", string);
 *         ssSetErrorStatus(S,msg);
 *         return;
 *      }
 *
 * See matlabroot/simulink/src/sfuntmpl_doc.c for more details.
 */

/*====================*
 * S-function methods *
 *====================*/

#define MDL_CHECK_PARAMETERS   /* Change to #undef to remove function */
#if defined(MDL_CHECK_PARAMETERS) && defined(MATLAB_MEX_FILE)
  /* Function: mdlCheckParameters =============================================
   * Abstract:
   *    mdlCheckParameters verifies new parameter settings whenever parameter
   *    change or are re-evaluated during a simulation. When a simulation is
   *    running, changes to S-function parameters can occur at any time during
   *    the simulation loop.
   */
static void mdlCheckParameters(SimStruct *S)
{
    if (PRM_TS(S) < 0)
        ssSetErrorStatus(S, "Sampling period Ts has to be positive.");
    if ((PRM_TCP_PORT(S) < 1) || (PRM_TCP_PORT(S) > 65536))
        ssSetErrorStatus(S, "TCP port must be in the range [1,65536]");
}
#endif /* MDL_CHECK_PARAMETERS */


/* Function: mdlInitializeSizes ===============================================
 * Abstract:
 *    The sizes information is used by Simulink to determine the S-function
 *    block's characteristics (number of inputs, outputs, states, etc.).
 */
static void mdlInitializeSizes(SimStruct *S)
{
    ssSetNumSFcnParams(S, PRM_COUNT);  /* Number of expected parameters */
    if (ssGetNumSFcnParams(S) != ssGetSFcnParamsCount(S)) {
        /* Return if number of expected != number of actual parameters */
        ssSetErrorStatus(S, "3-parameters required: Ts, IP address and TCP port number");
        return;
    }

  #if defined(MDL_CHECK_PARAMETERS) && defined(MATLAB_MEX_FILE)
    mdlCheckParameters(S);
    if (ssGetErrorStatus(S) != NULL) return;
  #endif
    
    ssSetNumContStates(S, 0);
    ssSetNumDiscStates(S, 0);

    // Set the number of input ports to one
    if (!ssSetNumInputPorts(S, 1)) return;
    ssSetInputPortWidth(S, 0, 56);
    ssSetInputPortDataType(S, 0, SS_UINT16);
    
    /*
     * Set direct feedthrough flag (1=yes, 0=no).
     * A port has direct feedthrough if the input is used in either
     * the mdlOutputs or mdlGetTimeOfNextVarHit functions.
     * See matlabroot/simulink/src/sfuntmpl_directfeed.txt.
     */

    // Set the number of output ports to zero
    if (!ssSetNumOutputPorts(S, 0)) return;
    
    ssSetNumSampleTimes(S, 1);
    ssSetNumRWork(S, 0);
    ssSetNumIWork(S, IWORK_COUNT);
    ssSetNumPWork(S, 0);
    ssSetNumModes(S, 0);
    ssSetNumNonsampledZCs(S, 0);

    /* Specify the sim state compliance to be same as a built-in block */
    ssSetSimStateCompliance(S, USE_DEFAULT_SIM_STATE);

    ssSetOptions(S, 0);
}

/* Function: mdlInitializeSampleTimes =========================================
 * Abstract:
 *    This function is used to specify the sample time(s) for your
 *    S-function. You must register the same number of sample times as
 *    specified in ssSetNumSampleTimes.
 */
static void mdlInitializeSampleTimes(SimStruct *S)
{
    ssSetSampleTime(S, 0, PRM_TS(S));
    ssSetOffsetTime(S, 0, 0.0);
}



#undef MDL_INITIALIZE_CONDITIONS   /* Change to #undef to remove function */
#if defined(MDL_INITIALIZE_CONDITIONS)
  /* Function: mdlInitializeConditions ========================================
   * Abstract:
   *    In this function, you should initialize the continuous and discrete
   *    states for your S-function block.  The initial states are placed
   *    in the state vector, ssGetContStates(S) or ssGetRealDiscStates(S).
   *    You can also perform any other initialization activities that your
   *    S-function may require. Note, this routine will be called at the
   *    start of simulation and if it is present in an enabled subsystem
   *    configured to reset states, it will be call when the enabled subsystem
   *    restarts execution to reset the states.
   */
static void mdlInitializeConditions(SimStruct *S)
{
}
#endif /* MDL_INITIALIZE_CONDITIONS */


#define MDL_START  /* Change to #undef to remove function */
#if defined(MDL_START)
  /* Function: mdlStart =======================================================
   * Abstract:
   *    This function is called once at start of model execution. If you
   *    have states that should be initialized once, this is the place
   *    to do it.
   */
static void mdlStart(SimStruct *S)
{
  #ifndef WITHOUT_HW
  // Initialize the TCP connection
  int sock;
 
  struct addrinfo hints, *addrs;  
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  
  char ipaddr[20];
  mxGetString(PRM_IPADDR(S), ipaddr, 20);
  
  char port[10];
  sprintf(port, "%d", (int)PRM_TCP_PORT(S));
 
  if (getaddrinfo(ipaddr, port, &hints, &addrs) != 0) {
    ssSetErrorStatus(S, "Generator - getaddr failed failed");
  }

  // Create the socket
  sock = socket(addrs->ai_family, addrs->ai_socktype, addrs->ai_protocol);     
  if ( sock < 0 ) {
      ssSetErrorStatus(S, "Generator - opening the TCP socket failed.");
  }

  // Connect to the socket
  if ( connect(sock, addrs->ai_addr, addrs->ai_addrlen) == -1 ) {
        ssSetErrorStatus(S, "Generator - failed to connect to the TCP server.");
  }

  IWORK_TCP_FD(S) = sock;

  #endif /*WITHOUT_HW*/
}
#endif /*  MDL_START */



/* Function: mdlOutputs =======================================================
 * Abstract:
 *    In this function, you compute the outputs of your S-function
 *    block.
 */
static void mdlOutputs(SimStruct *S, int_T tid)
{
    InputPtrsType inputPtr = ssGetInputPortSignalPtrs(S, 0);
    InputUInt16PtrsType phaseShiftsPtr = (InputUInt16PtrsType)inputPtr;
    
    #ifndef WITHOUT_HW
    
    char buf[1+2*56]; // one byte for the command type and 2*56 bytes for the phase shifts
    uint16_t* buff_phaseShifts = (uint16_t*)(buf+1);
    int_T sock = IWORK_TCP_FD(S);

    buf[0] = 'p';
    for(int i=0; i<56; ++i) {
        buff_phaseShifts[i] = *phaseShiftsPtr[i];
    }
    
    // send the phase shifts
    if(send(sock, buf, 1+2*56, 0) == -1){
        ssSetErrorStatus(S, "Failed to send the phase shifts to the generator.");
    }

    #endif /*WITHOUT_HW*/
}

#undef MDL_UPDATE  /* Change to #undef to remove function */
#if defined(MDL_UPDATE)
  /* Function: mdlUpdate ======================================================
   * Abstract:
   *    This function is called once for every major integration time step.
   *    Discrete states are typically updated here, but this function is useful
   *    for performing any tasks that should only take place once per
   *    integration step.
   */
static void mdlUpdate(SimStruct *S, int_T tid)
{
}
#endif /* MDL_UPDATE */

#undef MDL_DERIVATIVES  /* Change to #undef to remove function */
#if defined(MDL_DERIVATIVES)
  /* Function: mdlDerivatives =================================================
   * Abstract:
   *    In this function, you compute the S-function block's derivatives.
   *    The derivatives are placed in the derivative vector, ssGetdX(S).
   */
  static void mdlDerivatives(SimStruct *S)
  {
  }
#endif /* MDL_DERIVATIVES */



/* Function: mdlTerminate =====================================================
 * Abstract:
 *    In this function, you should perform any actions that are necessary
 *    at the termination of a simulation.  For example, if memory was
 *    allocated in mdlStart, this is the place to free it.
 */
  static void mdlTerminate(SimStruct *S)
  {
      #ifndef WITHOUT_HW
      int_T tcp_fd = IWORK_TCP_FD(S);
      // close the tcp port
      if (tcp_fd != -1){
          // disable the outputs of the generator
          if(send(tcp_fd, "d", 1, 0) == -1){
              ssSetErrorStatus(S, "Failed to send the disable command to the generator.");
          }
          close(tcp_fd);
      }
      IWORK_TCP_FD(S) = -1;
      #endif /*WITHOUT_HW*/
  }


/*======================================================*
 * See sfuntmpl_doc.c for the optional S-function methods *
 *======================================================*/

/*=============================*
 * Required S-function trailer *
 *=============================*/

#ifdef  MATLAB_MEX_FILE    /* Is this file being compiled as a MEX-file? */
#include "simulink.c"      /* MEX-file interface mechanism */
#else
#include "cg_sfun.h"       /* Code generation registration function */
#endif
