import { Home, Books, Smote, ZetaGAN, Summary } from './pages';
const rootPath = process.env.PUBLIC_URL;

const routes = [
  {
    path: `${rootPath}/`,
    component: Home,
    exact: true,
    breadcrumbName: 'Home'
  },
  {
    path: `${rootPath}/books`,
    component: Books,
    breadcrumbName: 'Generate Data'
  },
  {
    path: `${rootPath}/smote`,
    component: Smote,
    breadcrumbName: 'SMOTE'
  },
  {
    path: `${rootPath}/zetagan`,
    component: ZetaGAN,
    breadcrumbName: 'WGAN'
  },
  {
    path: `${rootPath}/summary`,
    component: Summary,
    breadcrumbName: 'Summary'
  }
];

export default routes;
export { rootPath };
