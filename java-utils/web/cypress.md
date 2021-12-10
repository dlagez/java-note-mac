### install

```shell
npm install cypress --save-dev
```

open

```shell
./node_modules/.bin/cypress open
```

ref : `https://docs.cypress.io/guides/getting-started/writing-your-first-test#Add-a-test-file`

## usage

### `cy.visit()`

We can pass the URL we want to visit to `cy.visit()`

### `cy.contains()`

To find this element by its contents, we'll use [cy.contains()](https://docs.cypress.io/api/commands/contains).

### `cy.contains('type').click()`

click on the link we found

### `cy.url().should('include', '/commands/actions')`

make an assertion about something on the new page we clicked into

### `cy.get()`

We can use [cy.get()](https://docs.cypress.io/api/commands/get) to select an element based on a CSS class.

### `.type()`

Then we can use the [.type()](https://docs.cypress.io/api/commands/type) command to enter text into the selected input

### `.should()`

we can verify that the value of the input reflects the text that was typed with another [.should()](https://docs.cypress.io/api/commands/should).



### official example

```js
describe('My First Test', ()=> {
    it('Visits the Kitchen Sink', ()=> {
        cy.visit('https://example.cypress.io')

        // 找到 type 这个单词，点击这个单词所在的连接
        cy.contains('type').click()
        // 跳转的连接里面应该包含 /commands/actions
        cy.url().should('include', '/commands/actions')
        // 根据css class选择元素
        cy.get('.action-email')
            // 在选择的元素上输入 文本
            .type('fake@email.com')
            // 检查输入的文本是否正确
            .should('have.value', 'fake@email.com')
    })
})
```

describe

```text
Visit: https://example.cypress.io
Find the element with content: type
Click on it
Get the URL
Assert it includes: /commands/actions
Get the input with the .action-email class
Type fake@email.com into the input
Assert the input reflects the new value
```



注意事项：

我们的测试跨越了两个页面，在第二个页面没有加载完成之前，测试代码会停止运行

### `cy.intercept(url)`

它用来发送请求的。

```js
// spying
cy.intercept('/users/**')
cy.intercept('GET', '/users*')
cy.intercept({
  method: 'GET',
  url: '/users*',
  hostname: 'localhost',
})

// spying and response stubbing
cy.intercept('POST', '/users*', {
  statusCode: 201,
  body: {
    name: 'Peter Pan',
  },
})

// spying, dynamic stubbing, request modification, etc.
cy.intercept('/users*', { hostname: 'localhost' }, (req) => {
  /* do something with request and/or response */
})
```

.get() 元素选择器

