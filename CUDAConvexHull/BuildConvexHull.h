#include "Point.h"
Point *FirstPoint = new Point(0,0);

int compare(const void * a, const void * b) {
    int result = 1;
	Point V1 = *(Point*) a;
	Point V2 = *(Point*) b;
    if ((V1.Subtract(*FirstPoint).VectorProduct(V2.Subtract(*FirstPoint)) < 0) ||
        ((V1.Subtract(*FirstPoint).VectorProduct(V2.Subtract(*FirstPoint)) == 0) &&
		(V1.Distance(*FirstPoint) < V2.Distance(*FirstPoint))))
        result = -1;
	if (V1.Equals(V2))
        result = 0;
    return result;
}

void Sorting(Point *arr, int N) {
	for (int i = 0 ; i < N; i++) {
		if (arr[i].Y > (*FirstPoint).Y)
            (*FirstPoint) = arr[i];
        if ((arr[i].Y == (*FirstPoint).Y) && (arr[i].X < (*FirstPoint).X))
            (*FirstPoint) = arr[i];
    }
	qsort(arr, N, sizeof(Point), compare);
}

void BuildHull(int *arr, int N,int *IntHull,int *NHull) {
	Point *Minutiae;
	int NoM;
	NoM = N;
	Minutiae = (Point*) malloc(NoM * sizeof(Point));
	Minutiae = (Point*) arr;
	Sorting(Minutiae,NoM);
	Point *Hull = (Point*) malloc (NoM *sizeof(Point));
	Hull[0] = Minutiae[0];
	Hull[1] = Minutiae[1];
	int top = 1;
	int nextToTop = 0;
    for (int i = 2; i < NoM; i++)    {
		while ((Minutiae[i].Subtract(Hull[nextToTop]).VectorProduct(Hull[top].Subtract(Hull[nextToTop])) <=0) && (!Hull[top].Equals(*FirstPoint))) {
			 top--;
			 if (Hull[top].Equals(*FirstPoint))
                 nextToTop = top;
             else
                 nextToTop = top-1;
         }
         top++;
		 nextToTop = top-1;
		 Hull[top] = Minutiae[i];
    }
	*NHull = top+1;
	IntHull = (int*)Hull;
}
