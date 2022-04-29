import uuid

import qrcode


def gen_qr(data, filename, error_correction):
    qr = qrcode.QRCode(
        version=1,
        error_correction=error_correction,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img.save(filename)


gen_qr("1", "1-L.png", qrcode.constants.ERROR_CORRECT_L)
gen_qr("1", "1-H.png", qrcode.constants.ERROR_CORRECT_H)

id_ = uuid.uuid4()
gen_qr(str(id_), "uuid-L.png", qrcode.constants.ERROR_CORRECT_L)
gen_qr(str(id_), "uuid-H.png", qrcode.constants.ERROR_CORRECT_H)
