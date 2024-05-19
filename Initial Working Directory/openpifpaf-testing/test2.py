#image_response = requests.get('https://raw.githubusercontent.com/vita-epfl/openpifpaf/master/docs/coco/000000081988.jpg')
#pil_im = PIL.Image.open(io.BytesIO(image_response.content)).convert('RGB')
pil_im = PIL.Image.open
im=np.asarray(pil_im)
