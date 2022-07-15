#include <Python.h>
#include "numpy/arrayobject.h"
#include "soundloc.h"
#include <stdint.h>

#if PY_MAJOR_VERSION < 3
#error "Requires Python 3"
#include "stopcompilation"
#endif

static PyObject* _filterAndSum(PyObject* self, PyObject* args) {
	PyArrayObject *gcc, *tdoas, *matout;
	//double **cin, **cout;
	double **cout;

	int n, m, P, K;
	int32_t T;

	if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &gcc, &PyArray_Type, &tdoas, &PyArray_Type, &matout)) return NULL;

	/* Object type checks */
	if (NULL == gcc) return NULL;
	if (NULL == tdoas) return NULL;
	if (NULL == matout) return NULL;
// insert type checks


	//Py_INCREF(matout);

	//cin = npArray2CPtrs(matin);
	cout = npArray2CPtrs(matout);

	n = gcc->dimensions[0];
	m = gcc->dimensions[1];
	T = gcc->dimensions[2];
	P = matout->dimensions[0];
	K = matout->dimensions[1];

//Calculation

	int32_t t;
	for (int p = 0; p < P; p++) {
		for (int k = 0; k < K; k++) {
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < m; j++) {
					t = *((int32_t *)PyArray_GETPTR3(tdoas, p, j, i));
					if (t < 0) {
						cout[p][k] += *((double *)PyArray_GETPTR4(gcc, i, j, T + t, k));
					} else {
						cout[p][k] += *((double *)PyArray_GETPTR4(gcc, i, j, t, k));
					}
				}
			}
		}
	}

	//freeCArray(cin);
	freeCArray(cout);
	Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
	{"_filterAndSum", &_filterAndSum, METH_VARARGS, "SRP-PHAT"},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef module_def = {
	PyModuleDef_HEAD_INIT,
	"soundloc",
	"SoundLoc backend C methods",
	-1,
	methods,
};

PyMODINIT_FUNC PyInit_soundloc() {
	PyObject* module = PyModule_Create(&module_def);
	if (module==NULL) return NULL;
	import_array();
	if (PyErr_Occurred()) return NULL;
	return module;
}

//Helper functions

double** npArray2CPtrs(PyArrayObject* arr) {
	double **c, *a;

	int n = arr->dimensions[0];
	int m = arr->dimensions[1];
	c = ptrArray(n);
	a = (double *) arr->data;

	for ( int i = 0; i < n; i++) {
		c[i] = a + i*m;
	}
	return c;
}

double** ptrArray(long n) {
	double** c;
	c = (double **)malloc((size_t) (n*sizeof(double)));
	if (!c) {
		printf("Allocation failed");
		exit(0);
	}
	return c;
}

void freeCArray(double** c) {
	free((char*) c);
}

//End of Helper functions
