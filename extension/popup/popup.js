// URL at which the API server is running
base_url = "http://10.194.29.215:80";

// Inject _getSelectedTextFromTab into current page and 
// populate the textarea for user input in the popup with the selected text
function getSelectedText() {
    // Get information about the currently active tab
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
        let tab = tabs[0];

        // Inject JavaScript into the active tab to get the text selected by the user
        chrome.scripting.executeScript(
            {
                target: { tabId: tab.id },              // Specify a target to inject JavaScript
                function: _getSelectedTextFromTab,      // Function to be injected into the target
            },
            ([res]) => {
            // If selection is not empty, populate the input textarea
                if (res["result"] !== "") {
                    document.getElementById("input_text").value = res["result"];
                }
            }
        );
    });
};

// Get the selected text from the current page
function _getSelectedTextFromTab() {
    var selection = window.getSelection().toString();
    return selection;
}


// Issue a POST request with a request body
function doPost(url, body, callback){
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() { 
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
            callback(xmlHttp.responseText);
    }
    xmlHttp.open("POST", url, true); // true for asynchronous 
    xmlHttp.setRequestHeader('Content-type', 'application/json; charset=utf-8');
    xmlHttp.send(body);
}


// Obtain the results from the API server
function getResults() {
    document.getElementById("results").style.display = "none";

    let error_box = document.getElementById("error_box");
    let text = document.getElementById("input_text").value;
    let loading_text = document.getElementById("loading_text");
    
    // If there is no input text, throw error 
    if (text == "") {
        error_box.innerHTML = "Please enter something!";
        error_box.style.display = "block";
    }
    else {
        error_box.style.display = "none";

        // Start displaying the spinner
        loading_text.innerHTML = "Summarizing...";
        document.getElementById("loading").style.display = "block";

        // Create the JSON request body as specified in the API endpoint
        var body = JSON.stringify({
            text: text
        })

        let predict_url = `${base_url}/predict`;     // POST endpoint to be hit

        doPost(predict_url, body, (res, err) => {
            // Stop displaying the spinner on receiving a response
            document.getElementById("loading").style.display = "none";
            if(err){
                error_box.innerHTML = "Error fetching the summary!";
                error_box.style.display = "block";
            }
            else {
                res = JSON.parse(res)
                // Populate the output textarea with the summarized text
                // document.getElementById("paraphrased_text").value = res.paraphrased.text;
                document.getElementById("summarized_text").value = res.text;
                
                // Display the output in the popup
                document.getElementById("results").style.display = "block";
            }
        })
    }
}


// Trigger the injection of script to get user selected text and 
// populate the input textarea whenever the DOM content is loaded
// (without waiting for images and stylesheets to finish loading)
document.addEventListener("DOMContentLoaded", getSelectedText);

// When the 'Process' button is clicked, send the POST request to the API server to obtain
// the result text and populate the results dynamically
document.getElementById("submit_text").addEventListener("click", getResults);