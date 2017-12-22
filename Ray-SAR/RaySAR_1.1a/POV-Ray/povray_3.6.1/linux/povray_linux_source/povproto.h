/****************************************************************************
 *                  povproto.h
 *
 * This module defines the prototypes for all system-independent functions.
 *
 * from Persistence of Vision(tm) Ray Tracer version 3.6.
 * Copyright 1991-2003 Persistence of Vision Team
 * Copyright 2003-2004 Persistence of Vision Raytracer Pty. Ltd.
 *---------------------------------------------------------------------------
 * NOTICE: This source code file is provided so that users may experiment
 * with enhancements to POV-Ray and to port the software to platforms other
 * than those supported by the POV-Ray developers. There are strict rules
 * regarding how you are permitted to use this file. These rules are contained
 * in the distribution and derivative versions licenses which should have been
 * provided with this file.
 *
 * These licences may be found online, linked from the end-user license
 * agreement that is located at http://www.povray.org/povlegal.html
 *---------------------------------------------------------------------------
 * This program is based on the popular DKB raytracer version 2.12.
 * DKBTrace was originally written by David K. Buck.
 * DKBTrace Ver 2.0-2.12 were written by David K. Buck & Aaron A. Collins.
 *---------------------------------------------------------------------------
 * $File: //depot/povray/3.6-release/source/povproto.h $
 * $Revision: #2 $
 * $Change: 2939 $
 * $DateTime: 2004/07/05 03:43:26 $
 * $Author: root $
 * $Log$
 *****************************************************************************/

#ifndef POVPROTO_H
#define POVPROTO_H

/* Prototypes for functions defined in mem.c */
#include "pov_mem.h"
#include "userio.h"

BEGIN_POV_NAMESPACE

/*****************************************************************************
* Global preprocessor defines
******************************************************************************/



/*****************************************************************************
* Global typedefs
******************************************************************************/



/*****************************************************************************
* Global variables
******************************************************************************/



/*****************************************************************************
* Global functions
******************************************************************************/

/* Prototypes for machine specific functions defined in "computer".c (ibm.c amiga.c unix.c etc.)*/
void display_finished (void);
int display_init (int width, int height);
void display_close (void);
void display_plot (int x, int y, unsigned char Red, unsigned char Green, unsigned char Blue, unsigned char Alpha);
void display_plot_rect (int x1, int x2, int y1, int y2,
  unsigned char Red, unsigned char Green, unsigned char Blue, unsigned char Alpha);

/* Prototypes for functions defined in userio.c */

int CDECL Debug_Info(const char *format, ...);

int CDECL Warning(unsigned int level, const char *format,...);
int CDECL WarningAt(unsigned int level, const char *filename, long line, unsigned long offset, const char *format, ...);
int CDECL Error(const char *format,...);
int CDECL PossibleError(const char *format,...);
int CDECL ErrorAt(const char *filename, long line, unsigned long offset, const char *format, ...);

/* Prototypes for functions defined in benchmark.c */

bool Write_Benchmark_File (const char *Scene_File_Name, const char *INI_File_Name) ;
unsigned int Get_Benchmark_Version (void) ;

END_POV_NAMESPACE

#endif
