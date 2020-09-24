import cv2


# Da como resultado el contorno con el area maxima dada una lista de contornos.
def get_greatest_contour(contours):
    max_contour = contours[0]
    for contour in contours:
        if cv2.contourArea(contour) > cv2.contourArea(max_contour):
            max_contour = contour

    return max_contour


# Printea en consola los valores de invariantes de Hu para el valor de area mayor como tambien el guardado actualmente.
def print_hu_moments(greatest_contour, saved_contours):
    if greatest_contour is not None and len(saved_contours) > 0:
        greatest_hu_moments = get_hu_moments(greatest_contour)
        saved_shape_hu_moments = get_hu_moments(saved_contours)

        print('Greatest Contour Hu Moments:\n{}'.format(greatest_hu_moments))
        print('Saved Contours Hu Moments:\n{}'.format(saved_shape_hu_moments))
    else:
        print("No se tiene un contorno guardado actualmente, para guardar uno presione F")


# Da como resultado el contorno con el area maxima dada una lista de contornos.
def get_greatest_contour(contours):
    max_contour = contours[0]
    for contour in contours:
        if cv2.contourArea(contour) > cv2.contourArea(max_contour):
            max_contour = contour

    return max_contour


# Calcula los momentos de Hu en base a un contorno
def get_hu_moments(contour):
    moments = cv2.moments(contour)
    return cv2.HuMoments(moments)


# Filtra los contornos que sean mayores a un max_area o menores a un min_area
def filter_contours(contours, min_area, max_area):
    return list(filter(lambda contour: min_area < cv2.contourArea(contour) < max_area, contours))
