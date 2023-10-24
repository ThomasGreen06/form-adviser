"use strict";

const upload = (elm) => {
    const file = elm.files[0];

    if(elm.files.length == 1) {
        const form_data = new FormData();
        form_data.set('file', file);

        fetch('/upload', {method: 'POST', body: form_data})
        .then(response => {
            if(!response.ok) {
                console.error('response.ok:', response.ok);
                console.error('esponse.status:', response.status);
                console.error('esponse.statusText:', response.statusText);
                throw new Error(response.statusText);
            }
            console.log("success");
            location.href = "/result";
        })
        .catch(error => {
            console.error(error);
        });
    }
}


const upload_model = (elm) => {
    const file = elm.files[0];

    if(elm.files.length == 1) {
        const form_data = new FormData();
        form_data.set('file', file);

        fetch('/upload_model', {method: 'POST', body: form_data})
        .then(response => {
            if(!response.ok) {
                console.error('response.ok:', response.ok);
                console.error('esponse.status:', response.status);
                console.error('esponse.statusText:', response.statusText);
                throw new Error(response.statusText);
            }
            console.log("success");
            location.href = "/upload";
        })
        .catch(error => {
            console.error(error);
        });
    }
}
