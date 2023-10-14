function predict() {
    var age = document.getElementById('age').value;
    var call_time = document.getElementById('call_time').value;
    var traffic = document.getElementById('traffic').value;
    var fee = document.getElementById('fee').value;

    var data = {
        "age": age,
        "call_time": call_time,
        "traffic": traffic,
        "fee": fee
    };

    fetch('http://127.0.0.1:5000/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById('prediction').innerText = '预测结果：' + data.predicted_label;
        })
        .catch(error => console.error('错误：', error));
}


// 生成箱线图
function generateBoxPlot() {
    fetch('/api/boxplot')
        .then(response => response.json())
        .then(data => {
            // Clear any previous boxplot
            const boxplotDiv = document.getElementById('boxplot');
            boxplotDiv.innerHTML = '';

            // Create an img element and set its source to the generated boxplot image
            const img = document.createElement('img');
            img.src = data.image_path;
            boxplotDiv.appendChild(img);
            // 调用动画函数
            animateBoxPlot();
        })
        .catch(error => console.error('Error generating boxplot:', error));
}

//
// function handleButtonClick() {
//     const button = document.getElementById('loadImagesButton');
//     button.addEventListener('click', () => {
//         // Load boxplot images and set up the carousel when the button is clicked
//         loadBoxplotImages();
//     });
// }
//
// // Call the function to handle the button click
// handleButtonClick();


// Attach an event listener to the button
const button = document.getElementById('loadImageButton');
// const button = document.querySelector('button');
button.addEventListener('click', generateBoxPlot);


function animateBoxPlot() {
    // 动画效果参数
    const animationParams = {
        targets: '.boxplot',  // 动画目标为所有箱线图元素
        // translateY: [50, 0],  // Y轴平移动画，从50px向上平移至初始位置
        // opacity: [0, 1],  // 透明度动画，从透明到不透明
        // easing: 'easeOutElastic',  // 使用弹性缓动效果
        // duration: 1200,  // 动画持续时间，单位为毫秒
        // delay: anime.stagger(200)  // 间隔200毫秒逐个显示

        //旋转
        // rotate: '1turn',
        // easing: 'easeInOutQuad',
        // duration: 1000

        //x平移
        //translateX: [200, 0],
        // easing: 'easeInOutQuad',
        // duration: 1000,

        //放大
        scale: [0, 1],
        easing: 'easeInOutQuad',
        duration: 1000

        //淡入淡出
        // opacity: [0, 1],
        // easing: 'easeInOutQuad',
        // duration: 2000
    };

    // 应用动画
    anime(animationParams);
}


// 每个特征对应的箱线图
async function loadBoxplotImages() {
    const features = ['age', 'call_time', 'traffic', 'fee'];
    const carousel = document.getElementById('boxplotCarousel');


    for (let i = 0; i < features.length; i++) {
        const feature = features[i];
        const response = await fetch(`/api/boxplot_image/${feature}`);
        const blob = await response.blob();

        const boxplotImage = document.createElement('img');
        boxplotImage.src = URL.createObjectURL(blob);
        boxplotImage.className = 'boxplot-image';

        carousel.appendChild(boxplotImage);

        // Add a delay of 1 second (1000 milliseconds) between each image
        const delay = i === features.length - 1 ? 0 : 700;  // No delay after the last image
        await new Promise(resolve => setTimeout(resolve, delay));
    }

    // Calculate the total width of the carousel
    const carouselWidth = features.length * 320;  // Assuming 10px margin and 300px width for each image
    carousel.style.width = `${carouselWidth}px`;
}


// 每个特征对应的箱线图
async function loadBoxplotImagesback() {
    const features = ['age', 'call_time', 'traffic', 'fee'];
    const carousel = document.getElementById('boxplotCarousel');

    for (const feature of features) {
        const response = await fetch(`/api/boxplot_image/${feature}`);
        const blob = await response.blob();

        const boxplotImage = document.createElement('img');
        boxplotImage.src = URL.createObjectURL(blob);
        boxplotImage.className = 'boxplot-image';
        carousel.appendChild(boxplotImage);
    }

    // Calculate the total width of the carousel
    const carouselWidth = features.length * 320;  // Assuming 10px margin and 300px width for each image
    carousel.style.width = `${carouselWidth}px`;
}

// Function to handle the button click
function handleButtonClick() {
    const button = document.getElementById('loadImagesButton');
    button.addEventListener('click', () => {
        // Load boxplot images and set up the carousel when the button is clicked
        loadBoxplotImages();
    });
}

// Call the function to handle the button click
handleButtonClick();



// 创建页码链接
function generatePaginationLinks(currentPage, totalPages) {
    const pagination = document.getElementById('pagination');
    pagination.innerHTML = '';

    // 表示在当前页码两侧要显示的相邻页码的数量
    const numAdjacentLinks = 2;  // Number of adjacent links to display
    const maxPage = Math.min(totalPages, currentPage + numAdjacentLinks);
    const minPage = Math.max(1, currentPage - numAdjacentLinks);

    // 创建表示特定页码的链接元素
    const createPageLink = (pageNumber) => {
        const link = document.createElement('a');
        link.href = '#';
        link.className = 'page-link';
        link.textContent = pageNumber;
        // 页码链接设置其事件监听器，以便在用户点击时调用 fetchData 函数加载对应的数据
        link.addEventListener('click', () => fetchData(pageNumber));
        return link;
    };

    // Add "Previous" button
    if (currentPage > 1) {
        const prevLink = createPageLink(currentPage - 1);
        prevLink.textContent = 'Previous';
        pagination.appendChild(prevLink);
    }

    // 循环生成当前页码附近的页码链接
    for (let i = minPage; i <= maxPage; i++) {
        pagination.appendChild(createPageLink(i));
    }

    // Add "Next" button
    if (currentPage < totalPages) {
        const nextLink = createPageLink(currentPage + 1);
        nextLink.textContent = 'Next';
        pagination.appendChild(nextLink);
    }
}

function fetchData(page) {
    const pageSize = 5;  // Adjust the page size as needed

    fetch(`http://127.0.0.1:5000/api/get_data?page=${page}&pageSize=${pageSize}`)
        .then(response => response.json())
        .then(data => {
            const tbody = document.getElementById('data-body');
            tbody.innerHTML = '';  // Clear existing data

            data.data.forEach(item => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${item.User_ID}</td>
                    <td>${item.age}</td>
                    <td>${item.call_time}</td>
                    <td>${item.traffic}</td>
                    <td>${item.fee}</td>
                    <td>${item.gender}</td>
                    <td>${item.package}</td>
                `;
                tbody.appendChild(row);
            });

            const totalPages = data.totalPages;  // 获取总页数
            console.log(totalPages);

            // Call the function to generate pagination links
            generatePaginationLinks(page, totalPages);
        })
        .catch(error => console.error('Error:', error));
}

// Fetch initial data and generate pagination links for page 1 on page load
fetchData(1);
