/*
 * S-function to support IRC inputs on RaspberryPi through IRC device
 *
 * Copyright (C) 2014 Pavel Pisa <pisa@cmp.felk.cvut.cz>
 *
 * Department of Control Engineering
 * Faculty of Electrical Engineering
 * Czech Technical University in Prague (CTU)
 *
 * The S-Function for ERT Linux can be distributed in compliance
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
 * Linux ERT code is available from
 *    http://rtime.felk.cvut.cz/gitweb/ert_linux.git
 * More CTU Linux target for Simulink components are available at
 *    http://lintarget.sourceforge.net/
 *
 * sfuntmpl_basic.c by The MathWorks, Inc. has been used to accomplish
 * required S-function structure.
 */


#define S_FUNCTION_NAME  sfTCPposition
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

#define PRM_TS(S)                   (mxGetScalar(ssGetSFcnParam(S, 0))) // Sampling period
#define PRM_IPADDR(S)               (ssGetSFcnParam(S, 1))              // IP address of the device
#define PRM_TCP_PORT(S)             (mxGetScalar(ssGetSFcnParam(S, 2))) // TCP port of the service on the device
#define PRM_NOBJS(S)                (mxGetScalar(ssGetSFcnParam(S, 3))) // Number of tracked objects
#define PRM_TB_POS(S)               (mxGetScalar(ssGetSFcnParam(S, 4))) // nonzero if positions in both color channels are to be read

#define PRM_COUNT                   5

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
    if (PRM_NOBJS(S) < 1)
        ssSetErrorStatus(S, "Number of objects must be at least one.");
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
        ssSetErrorStatus(S, "4-parameters required: Ts, IP address, TCP port number and number of objects");
        return;
    }

  #if defined(MDL_CHECK_PARAMETERS) && defined(MATLAB_MEX_FILE)
    mdlCheckParameters(S);
    if (ssGetErrorStatus(S) != NULL) return;
  #endif
    
    ssSetNumContStates(S, 0);
    ssSetNumDiscStates(S, 0);

    // Set the number of input ports to zero
    if (!ssSetNumInputPorts(S, 0)) return;
    
    /*
     * Set direct feedthrough flag (1=yes, 0=no).
     * A port has direct feedthrough if the input is used in either
     * the mdlOutputs or mdlGetTimeOfNextVarHit functions.
     * See matlabroot/simulink/src/sfuntmpl_directfeed.txt.
     */

    // Set the number of output ports to one if the objects are tracked
    // only in the green channel and to two if the objects are tracked
    // also in the red channel
    if (PRM_TB_POS(S)) {
      if (!ssSetNumOutputPorts(S, 2)) return;      
    } else {
      if (!ssSetNumOutputPorts(S, 1)) return;
    }
    
    // Set the parameters of the first output (positions in the green channel)
    ssSetOutputPortWidth(S, 0, 2*PRM_NOBJS(S));
    ssSetOutputPortDataType(S, 0, SS_UINT16);

    // Set the parameters of the seconds output (positions in the red channel)
    if (PRM_TB_POS(S)) {
      ssSetOutputPortWidth(S, 1, 2*PRM_NOBJS(S));
      ssSetOutputPortDataType(S, 1, SS_UINT16);
    }    

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
  #ifndef WITHOUT_HW
//     uint32_t irc_val_raw = 0;
//     int_T irc_dev_fd = IWORK_IRC_DEV_FD(S);
// 
//     if (irc_dev_fd == -1)
//         return;
// 
//     if (read(irc_dev_fd, &irc_val_raw, sizeof(uint32_t)) != sizeof(uint32_t)) {
//         ssSetErrorStatus(S, "/dev/ircX read failed");
//     }
// 
//     IWORK_IRC_ACT_VAL(S) = (int32_t)irc_val_raw;
//     if (PRM_RESET_AT_STARTUP(S)) {
//         IWORK_IRC_OFFSET(S) = -(int32_t)irc_val_raw;
//     } else {
//         IWORK_IRC_OFFSET(S) = 0;
//     }
  #endif /*WITHOUT_HW*/
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
//     int_T irc_dev_fd;
//     const char *irc_dev_name;

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
    ssSetErrorStatus(S, "getaddr failed failed");
  }

  // Create the socket
  sock = socket(addrs->ai_family, addrs->ai_socktype, addrs->ai_protocol);     
  if ( sock < 0 ) {
      ssSetErrorStatus(S, "Opening the TCP socket failed.");
  }

  // Connect to the socket
  if ( connect(sock, addrs->ai_addr, addrs->ai_addrlen) == -1 ) {
        ssSetErrorStatus(S, "Failed to connect to the TCP server.");
  }

  IWORK_TCP_FD(S) = sock;

  #endif /*WITHOUT_HW*/

    //mdlInitializeConditions(S);
}
#endif /*  MDL_START */



/* Function: mdlOutputs =======================================================
 * Abstract:
 *    In this function, you compute the outputs of your S-function
 *    block.
 */
static void mdlOutputs(SimStruct *S, int_T tid)
{
    uint16_T *y_G, *y_R; // output position in the green and red channels
    
    // Get the pointer to the first output
    y_G = ssGetOutputPortSignal(S,0);

    if(PRM_TB_POS(S))
      // Get the pointer to the second output (if needed)
      y_R = ssGetOutputPortSignal(S,1);
    
    #ifndef WITHOUT_HW
    #define BUFFERSIZE 100
    char buf[BUFFERSIZE];
    int numbytes;
    int_T sock = IWORK_TCP_FD(S);
    
    // send the request for the position
    if(send(sock, "tr", 2, 0) == -1){
        ssSetErrorStatus(S, "Failed to send the request for the position.");
    }
    // Receive the positions
    if ((numbytes=recv(sock, buf, BUFFERSIZE, 0)) == -1) {
        ssSetErrorStatus(S, "Failed to receive the positions.");
    }
    
    uint16_t N_tracked_objs_G = ((uint16_t*)buf)[0]; // Number of tracked objects in the green channel
    uint16_t N_tracked_objs_R = ((uint16_t*)buf)[1]; // Number of tracked objects in the red channel

    if(PRM_TB_POS(S) && N_tracked_objs_G != N_tracked_objs_R) {
      ssSetErrorStatus(S, "The numbers of tracked objects in green and red channel are not equal.");
    }

    if (N_tracked_objs_G < PRM_NOBJS(S)) {
        ssSetErrorStatus(S, "The number of tracked objects is smaller than the number of positions to be read.");
    }
    
    uint16_t *positions_G = (uint16_t*)(buf + 2*sizeof(uint16_t));
    
    // Copy the received positions to the output (green channel)
    for(int i=0;i<2*PRM_NOBJS(S);++i)
        y_G[i] = positions_G[i];

    // Copy the received positions to the output (red channel)
    if(PRM_TB_POS(S)) {
      uint16_t *positions_R = (uint16_t*)(buf + 2*sizeof(uint16_t) + N_tracked_objs_G*2*sizeof(uint16_t));
      for(int i=0;i<2*PRM_NOBJS(S);++i)
          y_R[i] = positions_R[i];
    }
    
    #else /*WITHOUT_HW*/
    for(int i=0;i<2*PRM_NOBJS(S);++i)
        y_G[i] = 0;
    if(PRM_TB_POS(S)) {
      for(int i=0;i<2*PRM_NOBJS(S);++i)
          y_R[i] = 1;
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
    // close the tcp port
    int_T tcp_fd = IWORK_TCP_FD(S);
    if (tcp_fd != -1)
      close(tcp_fd);    
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
