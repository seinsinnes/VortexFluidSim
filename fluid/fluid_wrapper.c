#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *fluid_accumulateVelocity(PyObject *self, PyObject *args);
static PyObject *fluid_setVortonStatsList(PyObject *self, PyObject *args);
static PyObject *fluid_computeVelocityAtPosition(PyObject *self, PyObject *args);

static char module_docstring[] =
    "This module provides math support for fluid simulation.";
static char computeVel_docstring[] =
    ".";

static PyMethodDef module_methods[] = {
    {"accumulateVelocity", fluid_accumulateVelocity, METH_VARARGS, computeVel_docstring},
    {"setVortonStatsList", fluid_setVortonStatsList, METH_VARARGS, computeVel_docstring},
    {"computeVelocityAtPosition", fluid_computeVelocityAtPosition,METH_VARARGS, computeVel_docstring},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initfluid(void)
{
    PyObject *m = Py_InitModule3("fluid", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
}

static PyObject *fluid_setVortonStatsList(PyObject *self, PyObject *args)
{
        PyObject *vortonStatsListObj;

  double **vortonStatsList;
  //Create C arrays from numpy objects:
  int typenum = NPY_DOUBLE;
  PyArray_Descr *descr;
  descr = PyArray_DescrFromType(typenum);
  npy_intp dims[3];

    PyArg_ParseTuple(args, "O", &vortonStatsListObj);

    PyArray_AsCArray(&vortonStatsListObj, (void **)&vortonStatsList, dims, 2, descr);
    /* Call the external C function to compute the chi-squared. */
    setVortonStatsList(vortonStatsList,dims[0]);

    /* Clean up. */
    //Py_DECREF(x_array);

    /*if (value < 0.0) {
        PyErr_SetString(PyExc_RuntimeError,
                    "Chi-squared returned an impossible value.");
        return NULL;
    }*/

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", 0);
    return ret;
}

static PyObject *fluid_computeVelocityAtPosition(PyObject *self, PyObject *args)
{

    PyObject *vPosQueryObj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "O", &vPosQueryObj))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    
    PyObject *vPosQuery_array = PyArray_FROM_OTF(vPosQueryObj, NPY_DOUBLE, NPY_IN_ARRAY);
    
    PyObject *vVelocity_array = PyArray_NewLikeArray(vPosQuery_array, NPY_ANYORDER, NULL, 0);                                      
    /* If that didn't work, throw an exception. */
    if (vVelocity_array == NULL || vPosQuery_array == NULL) {
        Py_XDECREF(vVelocity_array);
        Py_XDECREF(vPosQuery_array);
        return NULL;
    }

    /* Get pointers to the data as C-types. */
    double *vVelocity    = (double*)PyArray_DATA(vVelocity_array);
    double* vPosQuery    = (double*)PyArray_DATA(vPosQuery_array);

    /* Call the external C function to compute the chi-squared. */
    vVelocity[0] = 0.0;
    vVelocity[1] = 0.0;
    vVelocity[2] = 0.0;
    computeVelocityAtPosition(vVelocity,vPosQuery);

    /* Clean up. */
        Py_XDECREF(vPosQuery_array);

    /* Build the output tuple */
    //PyObject *ret = PyArray_SimpleNewFromData(int nd, npy_intp dims, int typenum, void* data)
    //PyObject *ret = Py_BuildValue("d", value);
    return vVelocity_array;
}

static PyObject *fluid_accumulateVelocity(PyObject *self, PyObject *args)
{
    double vortRadius;
    PyObject *vPosQueryObj, *vortPosObj, *vortVortObj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOOd", &vPosQueryObj, &vortPosObj, &vortVortObj,
                                        &vortRadius))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    
    PyObject *vPosQuery_array = PyArray_FROM_OTF(vPosQueryObj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *vortPos_array = PyArray_FROM_OTF(vortPosObj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *vortVort_array = PyArray_FROM_OTF(vortVortObj, NPY_DOUBLE, NPY_IN_ARRAY);
    
    PyObject *vVelocity_array = PyArray_NewLikeArray(vortVort_array, NPY_ANYORDER, NULL, 0);                                      
    /* If that didn't work, throw an exception. */
    if (vVelocity_array == NULL || vPosQuery_array == NULL || vortPos_array == NULL || vortVort_array == NULL) {
        //Py_XDECREF(vVelocity_array);
        Py_XDECREF(vPosQuery_array);
        Py_XDECREF(vortPos_array);
        Py_XDECREF(vortVort_array);
        return NULL;
    }

    /* Get pointers to the data as C-types. */
    double *vVelocity    = (double*)PyArray_DATA(vVelocity_array);
    double* vPosQuery    = (double*)PyArray_DATA(vPosQuery_array);
    double* vortPos = (double*)PyArray_DATA(vortPos_array);
    double* vortVort = (double*)PyArray_DATA(vortVort_array);

    /* Call the external C function to compute the chi-squared. */
    double value = accumulateVelocity(vVelocity,vPosQuery ,vortPos, vortVort, vortRadius );

    /* Clean up. */
        Py_XDECREF(vPosQuery_array);
        Py_XDECREF(vortPos_array);
        Py_XDECREF(vortVort_array);

    if (value < 0.0) {
        PyErr_SetString(PyExc_RuntimeError,
                    "ComputeVel returned an impossible value.");
        return NULL;
    }

    /* Build the output tuple */
    //PyObject *ret = PyArray_SimpleNewFromData(int nd, npy_intp dims, int typenum, void* data)
    //PyObject *ret = Py_BuildValue("d", value);
    return vVelocity_array;
}
